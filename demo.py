from original import *
import shutil, glob
from easyfuncs import download_from_url, CachedModels
model_library = CachedModels()

with gr.Blocks(title="🔊",theme=gr.themes.Base(primary_hue="rose",neutral_hue="zinc")) as app:
    with gr.Tabs():
        with gr.TabItem("Интерфейс"):
            with gr.Row():
                voice_model = gr.Dropdown(label="Модель голоса:", choices=sorted(names), value=lambda:sorted(names)[0] if len(sorted(names)) > 0 else '', interactive=True)
                refresh_button = gr.Button("Обновить", variant="primary")
                vc_transform0 = gr.Slider(
                    minimum=-20,
                    maximum=20,
                    step=1,
                    label="Тон",
                    value=0,
                    scale=2,
                    interactive=True,
                )
                but0 = gr.Button(value="🔊Преобразовать🔊", variant="primary")
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        dropbox = gr.File(label="Перетащите сюда аудиофайл и нажмите кнопку 'Обновить'")
                    with gr.Row():
                        paths_for_files = lambda path:[os.path.abspath(os.path.join(path, f)) for f in os.listdir(path) if os.path.splitext(f)[1].lower() in ('.mp3', '.wav', '.flac', '.ogg')]
                        input_audio0 = gr.Dropdown(
                            label="Путь к входному файлу:",
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
                        dropbox.upload(
                            fn=lambda audio:audio.name,
                            inputs=[dropbox],
                            outputs=[input_audio0])
                with gr.Column():
                    with gr.Accordion("Настройка index файла", open=False):
                        file_index2 = gr.Dropdown(
                            label="Index модели:",
                            choices=sorted(index_paths),
                            interactive=True,
                            value=sorted(index_paths)[0] if len(sorted(index_paths)) > 0 else ''
                        )
                        index_rate1 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label="Сила индекса",
                            value=0.66,
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
                vc_output1 = gr.Textbox(label="Информация", placeholder="Добро пожаловать!",visible=False)
                but0.click(
                    vc.vc_single,
                    [
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
                    outputs=[protect0, protect0, file_index2, file_index2],
                    api_name="infer_change_voice",
                )
        with gr.TabItem("Загрузка модели"):
            with gr.Row():
                url_input = gr.Textbox(label="URL модели:", value="",placeholder="https://...", scale=6)
                name_output = gr.Textbox(label="Сохранить как:", value="",placeholder="Shanin",scale=2)
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

    if config.iscolab:
        app.queue(concurrency_count=511, max_size=1022).launch(share=True)
    else:
        app.queue(concurrency_count=511, max_size=1022).launch(
            server_name="0.0.0.0",
            inbrowser=not config.noautoopen,
            server_port=config.listen_port,
            quiet=True,
        )
