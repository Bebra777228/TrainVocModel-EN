import logging
import os
import sys
import warnings

import absl.logging

# Настройка уровня логирования для различных библиотек
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.WARNING)
logging.getLogger("h5py").setLevel(logging.WARNING)
logging.getLogger("jax").setLevel(logging.WARNING)
logging.getLogger("numexpr").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
absl.logging.set_verbosity(absl.logging.WARNING)

# Подавление предупреждений
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

now_dir = os.getcwd()
sys.path.append(os.path.join(now_dir))

import datetime
from random import randint
from time import sleep
from time import time as ttime

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from rvc.lib.algorithm.commons import clip_grad_value_, slice_segments
from rvc.lib.algorithm.models import MultiPeriodDiscriminatorV2 as Discriminator
from rvc.lib.algorithm.models import SynthesizerTrnMs768NSFsid as Synthesizer
from rvc.train.data_utils import DistributedBucketSampler as Sampler
from rvc.train.data_utils import TextAudioCollateMultiNSFsid as AudioCollate
from rvc.train.data_utils import TextAudioLoaderMultiNSFsid as AudioLoader
from rvc.train.extract.extract_model import extract_model
from rvc.train.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from rvc.train.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from rvc.train.utils import get_hparams, get_logger, latest_checkpoint_path, load_checkpoint, save_checkpoint, summarize

hps = get_hparams()
os.environ["CUDA_VISIBLE_DEVICES"] = hps.gpus.replace("-", ",")
n_gpus = len(hps.gpus.split("-"))

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

global_step = 0


class EpochRecorder:
    def __init__(self):
        self.last_time = ttime()

    def record(self):
        now_time = ttime()
        elapsed_time = now_time - self.last_time
        self.last_time = now_time
        elapsed_time = round(elapsed_time, 1)
        elapsed_time_str = str(datetime.timedelta(seconds=int(elapsed_time)))
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        return f"Скорость: [{elapsed_time_str}] | Время: [{current_time}]"


def main():
    n_gpus = torch.cuda.device_count()

    if not torch.cuda.is_available():
        print("NO GPU DETECTED: falling back to CPU - this may take a while")
        n_gpus = 1

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(20000, 55555))

    children = []
    logger = get_logger(hps.model_dir)
    for i in range(n_gpus):
        subproc = mp.Process(
            target=run,
            args=(i, n_gpus, hps, logger),
        )
        children.append(subproc)
        subproc.start()

    for i in range(n_gpus):
        children[i].join()


def run(rank, n_gpus, hps, logger: logging.Logger):
    global global_step
    if rank == 0:
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    backend = "gloo" if os.name == "nt" or not torch.cuda.is_available() else "nccl"
    if rank == 0:
        logger.info(f"Используемый бэкенд распределенных вычислений: {backend}")
    try:
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            world_size=n_gpus,
            rank=rank,
        )
    except:
        dist.init_process_group(
            backend=backend,
            init_method="env://?use_libuv=False",
            world_size=n_gpus,
            rank=rank,
        )

    torch.manual_seed(hps.train.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    collate_fn = AudioCollate()
    train_dataset = AudioLoader(hps.data.training_files, hps.data)
    train_sampler = Sampler(
        train_dataset,
        hps.train.batch_size * n_gpus,
        [100, 200, 300, 400, 500, 600, 700, 800, 900],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )
    train_loader = DataLoader(
        train_dataset,
        num_workers=2,  # 4
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=True,
        prefetch_factor=8,
    )

    net_g = Synthesizer(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
        is_half=hps.train.fp16_run,
        sr=hps.sample_rate,
    )
    net_d = Discriminator(hps.model.use_spectral_norm)

    if torch.cuda.is_available():
        net_g = net_g.cuda(rank)
        net_d = net_d.cuda(rank)

    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )

    if torch.cuda.is_available():
        net_g = DDP(net_g, device_ids=[rank])
        net_d = DDP(net_d, device_ids=[rank])
    else:
        net_g = DDP(net_g)
        net_d = DDP(net_d)

    try:
        _, _, _, epoch_str = load_checkpoint(latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d)
        _, _, _, epoch_str = load_checkpoint(latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g)
        epoch_str += 1
        global_step = (epoch_str - 1) * len(train_loader)
    except:
        epoch_str = 1
        global_step = 0
        if hps.pretrainG != "":
            if rank == 0:
                logger.info("loaded pretrained %s" % (hps.pretrainG))
            if hasattr(net_g, "module"):
                logger.info(net_g.module.load_state_dict(torch.load(hps.pretrainG, map_location="cpu", weights_only=True)["model"]))
            else:
                logger.info(net_g.load_state_dict(torch.load(hps.pretrainG, map_location="cpu", weights_only=True)["model"]))
        if hps.pretrainD != "":
            if rank == 0:
                logger.info("loaded pretrained %s" % (hps.pretrainD))
            if hasattr(net_d, "module"):
                logger.info(net_d.module.load_state_dict(torch.load(hps.pretrainD, map_location="cpu", weights_only=True)["model"]))
            else:
                logger.info(net_d.load_state_dict(torch.load(hps.pretrainD, map_location="cpu", weights_only=True)["model"]))

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)

    scaler = GradScaler(enabled=hps.train.fp16_run)

    cache = []
    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, None],
                logger,
                [writer_eval],
                cache,
            )
        else:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, None],
                None,
                None,
                cache,
            )
        scheduler_g.step()
        scheduler_d.step()


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers, cache):
    net_g, net_d = nets
    optim_g, optim_d = optims
    train_loader, eval_loader = loaders
    if writers is not None:
        writer = writers[0]

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()

    data_iterator = enumerate(train_loader)
    epoch_recorder = EpochRecorder()
    for batch_idx, info in data_iterator:
        phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, wave_lengths, sid = info
        if torch.cuda.is_available():
            phone = phone.cuda(rank, non_blocking=True)
            phone_lengths = phone_lengths.cuda(rank, non_blocking=True)
            pitch = pitch.cuda(rank, non_blocking=True)
            pitchf = pitchf.cuda(rank, non_blocking=True)
            sid = sid.cuda(rank, non_blocking=True)
            spec = spec.cuda(rank, non_blocking=True)
            spec_lengths = spec_lengths.cuda(rank, non_blocking=True)
            wave = wave.cuda(rank, non_blocking=True)

        with autocast(device_type="cuda", enabled=hps.train.fp16_run):
            (
                y_hat,
                ids_slice,
                x_mask,
                z_mask,
                (z, z_p, m_p, logs_p, m_q, logs_q),
            ) = net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            y_mel = slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
            with autocast(device_type="cuda", enabled=False):
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.float().squeeze(1),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
            if hps.train.fp16_run == True:
                y_hat_mel = y_hat_mel.half()
            wave = slice_segments(wave, ids_slice * hps.data.hop_length, hps.train.segment_size)

            y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
            with autocast(device_type="cuda", enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
        optim_d.zero_grad()
        scaler.scale(loss_disc).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(device_type="cuda", enabled=hps.train.fp16_run):
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
            with autocast(device_type="cuda", enabled=False):
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        global_step += 1

    if rank == 0 and epoch % hps.train.log_interval == 0:
        if loss_mel > 75:
            loss_mel = 75
        if loss_kl > 9:
            loss_kl = 9

        scalar_dict = {
            "grad/norm_d": grad_norm_d,
            "grad/norm_g": grad_norm_g,
            "loss/g/total": loss_gen_all,
            "loss/d/total": loss_disc,
            "loss/g/fm": loss_fm,
            "loss/g/mel": loss_mel,
            "loss/g/kl": loss_kl,
        }
        summarize(writer=writer, epoch=epoch, scalars=scalar_dict)

    if epoch % hps.save_every_epoch == 0 and rank == 0:
        save_checkpoint(
            net_g,
            optim_g,
            hps.train.learning_rate,
            epoch,
            os.path.join(hps.model_dir, "G_{}.pth".format(2333333)),
        )
        save_checkpoint(
            net_d,
            optim_d,
            hps.train.learning_rate,
            epoch,
            os.path.join(hps.model_dir, "D_{}.pth".format(2333333)),
        )

        if hasattr(net_g, "module"):
            ckpt = net_g.module.state_dict()
        else:
            ckpt = net_g.state_dict()
        logger.info(
            "Сохранение модели - %s_e%s_s%s: %s"
            % (
                hps.name,
                epoch,
                global_step,
                extract_model(
                    ckpt,
                    hps.sample_rate,
                    hps.name + "_e%s_s%s" % (epoch, global_step),
                    epoch,
                    hps,
                ),
            )
        )

    if rank == 0:
        logger.info("====> Эпоха: {}/{} | Шаг: {} | {}".format(epoch, hps.total_epoch, global_step, epoch_recorder.record()))

    if epoch >= hps.total_epoch and rank == 0:
        logger.info("Тренировка успешно завершена. Завершение программы...")

        if hasattr(net_g, "module"):
            ckpt = net_g.module.state_dict()
        else:
            ckpt = net_g.state_dict()
        logger.info("Финальная модель успешно сохранена: %s" % (extract_model(ckpt, hps.sample_rate, hps.name, epoch, hps)))
        sleep(1)
        os._exit(2333333)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
