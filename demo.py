from original import *
import shutil, glob
from easyfuncs import download_from_url, CachedModels
os.makedirs("dataset",exist_ok=True)
model_library = CachedModels()

with gr.Blocks(title="🔊",theme=gr.themes.Base(primary_hue="rose",neutral_hue="zinc")) as app:
    with gr.Tabs():
        with gr.TabItem("Интерфейс"):
            with gr.Row():
                voice_model = gr.Dropdown(label="Модель голоса", choices=sorted(names), value=lambda:sorted(names)[0] if len(sorted(names)) > 0 else '', interactive=True)
                refresh_button = gr.Button("Обновить", variant="primary")
                spk_item = gr.Slider(
                    minimum=0,
                    maximum=2333,
                    step=1,
                    label="Идентификатор спикера",
                    value=0,
                    visible=False,
                    interactive=True,
                )
                vc_transform0 = gr.Number(
                    label="Тон",
                    value=0
                )
                but0 = gr.Button(value="Преобразовать", variant="primary")
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        dropbox = gr.File(label="Перетащите сюда аудиофайл и нажмите кнопку 'Обновить'.")
                    with gr.Row():
                        record_button=gr.Audio(source="microphone", label="Записать звук с микрофона.", type="filepath")
                    with gr.Row():
                        paths_for_files = lambda path:[os.path.abspath(os.path.join(path, f)) for f in os.listdir(path) if os.path.splitext(f)[1].lower() in ('.mp3', '.wav', '.flac', '.ogg')]
                        input_audio0 = gr.Dropdown(
                            label="Путь к входному файлу",
                            value=paths_for_files('audios')[0] if len(paths_for_files('audios')) > 0 else '',
                            choices=paths_for_files('audios'), # Показывать только абсолютные пути к аудиофайлам с расширениями .mp3, .wav, .flac или .ogg
                            allow_custom_value=True
                        )
                    with gr.Row():
                        audio_player = gr.Audio()
                        input_audio0.change(
                            inputs=[input_audio0],
                            outputs=[audio_player],
                            fn=lambda path: {"value":path,"__type__":"update"} if os.path.exists(path) else None
                        )
                        record_button.stop_recording(
                            fn=lambda audio:audio, #TODO сохранить wav lambda
                            inputs=[record_button],
                            outputs=[input_audio0])
                        dropbox.upload(
                            fn=lambda audio:audio.name,
                            inputs=[dropbox],
                            outputs=[input_audio0])
                with gr.Column():
                    with gr.Accordion("Изменить индекс", open=False):
                        file_index2 = gr.Dropdown(
                            label="Изменить индекс",
                            choices=sorted(index_paths),
                            interactive=True,
                            value=sorted(index_paths)[0] if len(sorted(index_paths)) > 0 else ''
                        )
                        index_rate1 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label="Сила индекса",
                            value=0.5,
                            interactive=True,
                        )
                    vc_output2 = gr.Audio(label="Выход")
                    with gr.Accordion("Дополнительные настройки", open=False):
                        f0method0 = gr.Radio(
                            label="Метод",
                            choices=["pm", "harvest", "crepe", "rmvpe"]
                            if config.dml == False
                            else ["pm", "harvest", "rmvpe"],
                            value="rmvpe",
                            interactive=True,
                        )
                        filter_radius0 = gr.Slider(
                            minimum=0,
                            maximum=7,
                            label="Снижение шума дыхания (только для Harvest)",
                            value=3,
                            step=1,
                            interactive=True,
                        )
                        resample_sr0 = gr.Slider(
                            minimum=0,
                            maximum=48000,
                            label="Перевыборка",
                            value=0,
                            step=1,
                            interactive=True,
                            visible=False
                        )
                        rms_mix_rate0 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label="Нормализация громкости",
                            value=0,
                            interactive=True,
                        )
                        protect0 = gr.Slider(
                            minimum=0,
                            maximum=0.5,
                            label="Защита от шума дыхания (0 - включено, 0.5 - выключено)",
                            value=0.33,
                            step=0.01,
                            interactive=True,
                        )
                        if voice_model != None: vc.get_vc(voice_model.value,protect0,protect0)
                    file_index1 = gr.Textbox(
                        label="Путь к индексному файлу",
                        interactive=True,
                        visible=False#Здесь не используется
                    )
                    refresh_button.click(
                        fn=change_choices,
                        inputs=[],
                        outputs=[voice_model, file_index2],
                        api_name="infer_refresh",
                    )
                    refresh_button.click(
                        fn=lambda:{"choices":paths_for_files('audios'),"__type__":"update"}, #TODO проверить, правильно ли возвращается отсортированный список аудиофайлов в папке 'audios' с расширениями '.wav', '.mp3', '.ogg' или '.flac'
                        inputs=[],
                        outputs = [input_audio0],
                    )
                    refresh_button.click(
                        fn=lambda:{"value":paths_for_files('audios')[0],"__type__":"update"} if len(paths_for_files('audios')) > 0 else {"value":"","__type__":"update"}, #TODO проверить, правильно ли возвращается отсортированный список аудиофайлов в папке 'audios' с расширениями '.wav', '.mp3', '.ogg' или '.flac'
                        inputs=[],
                        outputs = [input_audio0],
                    )
            with gr.Row():
                f0_file = gr.File(label="Путь к файлу F0", visible=False)
            with gr.Row():
                vc_output1 = gr.Textbox(label="Информация", placeholder="Добро пожаловать!",visible=True)
                but0.click(
                    vc.vc_single,
                    [
                        spk_item,
                        input_audio0,
                        vc_transform0,
                        f0_file,
                        f0method0,
                        file_index1,
                        file_index2,
                        index_rate1,
                        filter_radius0,
                        resample_sr0,
                        rms_mix_rate0,
                        protect0,
                    ],
                    [vc_output1, vc_output2],
                    api_name="infer_convert",
                )
                voice_model.change(
                    fn=vc.get_vc,
                    inputs=[voice_model, protect0, protect0],
                    outputs=[spk_item, protect0, protect0, file_index2, file_index2],
                    api_name="infer_change_voice",
                )
        with gr.TabItem("Загрузка модели"):
            with gr.Row():
                url_input = gr.Textbox(label="URL модели", value="",placeholder="https://...", scale=6)
                name_output = gr.Textbox(label="Сохранить как", value="",placeholder="MyModel",scale=2)
                url_download = gr.Button(value="Загрузить модель",scale=2)
                url_download.click(
                    inputs=[url_input,name_output],
                    outputs=[url_input],
                    fn=download_from_url,
                )
            with gr.Row():
                model_browser = gr.Dropdown(choices=list(model_library.models.keys()),label="Пользовательские модели",scale=5)
                download_from_browser = gr.Button(value="Получить",scale=2)
                download_from_browser.click(
                    inputs=[model_browser],
                    outputs=[model_browser],
                    fn=lambda model: download_from_url(model_library.models[model],model),
                )
        with gr.TabItem("Тренировка"):
            with gr.Row():
                with gr.Column():
                    training_name = gr.Textbox(label="Дайте имя своей модели", value="My-Voice",placeholder="My-Voice")
                    np7 = gr.Slider(
                        minimum=0,
                        maximum=config.n_cpu,
                        step=1,
                        label="Количество процессов CPU, используемых для выделения характеристик тона",
                        value=int(np.ceil(config.n_cpu / 1.5)),
                        interactive=True,
                    )
                    sr2 = gr.Radio(
                        label="Частота дискретизации",
                        choices=["40k", "32k"],
                        value="40k",
                        interactive=True,
                        visible=False
                    )
                    if_f0_3 = gr.Radio(
                        label="Будет ли ваша модель использоваться для пения? Если нет, вы можете проигнорировать это.",
                        choices=[True, False],
                        value=True,
                        interactive=True,
                        visible=False
                    )
                    version19 = gr.Radio(
                        label="Версия",
                        choices=["v1", "v2"],
                        value="v2",
                        interactive=True,
                        visible=False,
                    )
                    dataset_folder = gr.Textbox(
                        label="Папка с набором данных", value='dataset'
                    )
                    easy_uploader = gr.Files(label="Перетащите сюда ваши аудиофайлы",file_types=['audio'])
                    but1 = gr.Button("1. Обработать", variant="primary")
                    info1 = gr.Textbox(label="Информация", value="",visible=True)
                    easy_uploader.upload(inputs=[dataset_folder],outputs=[],fn=lambda folder:os.makedirs(folder,exist_ok=True))
                    easy_uploader.upload(
                        fn=lambda files,folder: [shutil.copy2(f.name,os.path.join(folder,os.path.split(f.name)[1])) for f in files] if folder != "" else gr.Warning('Пожалуйста, укажите имя папки для вашего набора данных'),
                        inputs=[easy_uploader, dataset_folder],
                        outputs=[])
                    gpus6 = gr.Textbox(
                        label="Введите номера GPU через дефис, (например, 0-1-2)",
                        value=gpus,
                        interactive=True,
                        visible=F0GPUVisible,
                    )
                    gpu_info9 = gr.Textbox(
                        label="Информация о GPU", value=gpu_info, visible=F0GPUVisible
                    )
                    spk_id5 = gr.Slider(
                        minimum=0,
                        maximum=4,
                        step=1,
                        label="Идентификатор спикера",
                        value=0,
                        interactive=True,
                        visible=False
                    )
                    but1.click(
                        preprocess_dataset,
                        [dataset_folder, training_name, sr2, np7],
                        [info1],
                        api_name="train_preprocess",
                    )
                with gr.Column():
                    f0method8 = gr.Radio(
                        label="Метод выделения характеристик F0",
                        choices=["pm", "harvest", "dio", "rmvpe", "rmvpe_gpu"],
                        value="rmvpe_gpu",
                        interactive=True,
                    )
                    gpus_rmvpe = gr.Textbox(
                        label="Номера GPU для использования через дефис (например 0-1-2)",
                        value="%s-%s" % (gpus, gpus),
                        interactive=True,
                        visible=F0GPUVisible,
                    )
                    but2 = gr.Button("2. Выделить характеристики", variant="primary")
                    info2 = gr.Textbox(label="Информация", value="", max_lines=8)
                    f0method8.change(
                        fn=change_f0_method,
                        inputs=[f0method8],
                        outputs=[gpus_rmvpe],
                    )
                    but2.click(
                        extract_f0_feature,
                        [
                            gpus6,
                            np7,
                            f0method8,
                            if_f0_3,
                            training_name,
                            version19,
                            gpus_rmvpe,
                        ],
                        [info2],
                        api_name="train_extract_f0_feature",
                    )
                with gr.Column():
                    total_epoch11 = gr.Slider(
                        minimum=2,
                        maximum=2000,
                        step=1,
                        label="Эпохи (больше эпох может улучшить качество, но занимает больше времени)",
                        value=300,
                        interactive=True,
                    )
                    but4 = gr.Button("3. Тренировать индекс", variant="primary")
                    but3 = gr.Button("4. Тренировать модель", variant="primary")
                    info3 = gr.Textbox(label="Информация", value="", max_lines=10)
                    with gr.Accordion(label="Общие настройки", open=False):
                        gpus16 = gr.Textbox(
                            label="GPU через дефис (например 0-1-2)",
                            value="0",
                            interactive=True,
                            visible=True
                        )
                        save_epoch10 = gr.Slider(
                            minimum=1,
                            maximum=50,
                            step=1,
                            label="Частота сохранения модели",
                            value=25,
                            interactive=True,
                        )
                        batch_size12 = gr.Slider(
                            minimum=1,
                            maximum=20,
                            step=1,
                            label="Размер пакета",
                            value=default_batch_size,
                            interactive=True,
                        )
                        if_save_latest13 = gr.Radio(
                            label="Сохранять только последнюю модель",
                            choices=["yes", "no"],
                            value="yes",
                            interactive=True,
                            visible=False
                        )
                        if_cache_gpu17 = gr.Radio(
                            label="Если ваш набор данных МЕНЬШЕ 10 минут, кэшируйте его для более быстрой тренировки",
                            choices=["yes", "no"],
                            value="yes",
                            interactive=True,
                        )
                        if_save_every_weights18 = gr.Radio(
                            label="Сохранять маленькую модель после каждого сохранения",
                            choices=["yes", "no"],
                            value="yes",
                            interactive=True,
                        )
                        with gr.Accordion(label="Список предварительно обученных моделей", open=False):
                            pretrained = lambda sr, letter: [os.path.abspath(os.path.join('assets/pretrained_v2', file)) for file in os.listdir('assets/pretrained_v2') if file.endswith('.pth') and sr in file and letter in file]
                            pretrained_G14 = gr.Dropdown(
                                label="Предварительно обученная модель G",
                                # Получить список всех предобученных моделей G в assets/pretrained_v2, заканчивающихся на .pth
                                choices = pretrained(sr2.value, 'G'),
                                value=pretrained(sr2.value, 'G')[0] if len(pretrained(sr2.value, 'G')) > 0 else '',
                                interactive=True,
                                visible=True
                            )
                            pretrained_D15 = gr.Dropdown(
                                label="Предварительно обученная модель D",
                                choices = pretrained(sr2.value, 'D'),
                                value= pretrained(sr2.value, 'D')[0] if len(pretrained(sr2.value, 'G')) > 0 else '',
                                visible=True,
                                interactive=True
                            )
                    with gr.Row():
                        download_model = gr.Button('5. Скачать файлы модели')
                    with gr.Row():
                        model_files = gr.Files(label='Ваша модель и индексный файл могут быть загружены здесь:')
                        download_model.click(
                            fn=lambda name: os.listdir(f'assets/weights/{name}') + glob.glob(f'logs/{name.split(".")[0]}/added_*.index'),
                            inputs=[training_name],
                            outputs=[model_files, info3])
                    with gr.Row():
                        sr2.change(
                            change_sr2,
                            [sr2, if_f0_3, version19],
                            [pretrained_G14, pretrained_D15],
                        )
                        version19.change(
                            change_version19,
                            [sr2, if_f0_3, version19],
                            [pretrained_G14, pretrained_D15, sr2],
                        )
                        if_f0_3.change(
                            change_f0,
                            [if_f0_3, sr2, version19],
                            [f0method8, pretrained_G14, pretrained_D15],
                        )
                    with gr.Row():
                        but5 = gr.Button("Тренировка в один клик", variant="primary", visible=True)
                        but3.click(
                            click_train,
                            [
                                training_name,
                                sr2,
                                if_f0_3,
                                spk_id5,
                                save_epoch10,
                                total_epoch11,
                                batch_size12,
                                if_save_latest13,
                                pretrained_G14,
                                pretrained_D15,
                                gpus16,
                                if_cache_gpu17,
                                if_save_every_weights18,
                                version19,
                            ],
                            info3,
                            api_name="train_start",
                        )
                        but4.click(train_index, [training_name, version19], info3)
                        but5.click(
                            train1key,
                            [
                                training_name,
                                sr2,
                                if_f0_3,
                                dataset_folder,
                                spk_id5,
                                np7,
                                f0method8,
                                save_epoch10,
                                total_epoch11,
                                batch_size12,
                                if_save_latest13,
                                pretrained_G14,
                                pretrained_D15,
                                gpus16,
                                if_cache_gpu17,
                                if_save_every_weights18,
                                version19,
                                gpus_rmvpe,
                            ],
                            info3,
                            api_name="train_start_all",
                        )

    if config.iscolab:
        app.queue(concurrency_count=511, max_size=1022).launch(share=True)
    else:
        app.queue(concurrency_count=511, max_size=1022).launch(
            server_name="0.0.0.0",
            inbrowser=not config.noautoopen,
            server_port=config.listen_port,
            quiet=True,
        )
