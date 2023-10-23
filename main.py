import logging
logging.basicConfig(level=logging.ERROR)

from rvc.infer import infer
import os
import sys
import json
import torch


class CLI_Interface:

    def __init__(self) -> None:
        self.pitch_adjustment = None
        self.model_path = ""
        self.index_file = ""
        self.device = ""
        self.pitch_extraction_method = ""
        self.audio_file = ""
        
    def load_settings(self):
        with open("settings.json", "r") as file:
            settings = json.load(file)
        
        if "pitch_adjustment" in settings:
            if isinstance(settings["pitch_adjustment"], int) and -12 <= settings["pitch_adjustment"] <= 12:
                self.pitch_adjustment = settings["pitch_adjustment"]
            else:
                print(f"Invalid pitch adjustment {settings['pitch_adjustment']} in settings.")

        if "pitch_extraction_method" in settings:
            if settings["pitch_extraction_method"] in ["pm", "harvest", "crepe", "rmvpe"]:
                self.pitch_extraction_method = settings["pitch_extraction_method"]
            else:
                print(f"Invalid pitch extraction method {settings['pitch_extraction_method']} in settings.")
            
        if "model" in settings:
            if isinstance(settings["model"], str):
                model = ""
                for root, _, files in os.walk("models"):
                    for file in files:
                        if file.endswith(settings["model"] + ".pth"):
                            model = os.path.join(root, file)
                if model != "":
                    matching_index_files = []
                    for file in os.listdir(os.path.join("models", "index")):
                        if os.path.isfile(os.path.join("models", "index", file)):
                            if file.endswith(f"{settings['model']}_v1.index") or file.endswith(f"{settings['model']}_v2.index") or file.endswith(f"{settings['model']}.index"):
                                matching_index_files.append(file)
                    if len(matching_index_files) == 1:
                        self.model_path = model
                        self.index_file = os.path.join("models", "index", matching_index_files[0])
                    elif len(matching_index_files) == 0:
                        print(f"The index for the model {settings['model']} could not be determined.")
                        sys.exit(1)
                    else:
                        print(f"Found multiple matching indexes {matching_index_files} for model {settings['model']}.")
                        sys.exit(1)
                else:
                    print(f"Model file for model with name {settings['model']} could not be found.")
            else:
                print(f"Field 'model' in settings needs to be a string.")
        
        if "device" in settings:
            if isinstance(settings["device"], str):
                if settings["device"] == "cpu":
                    self.device = settings["device"]
                elif ":" in settings["device"]:
                    parts = settings["device"].split(":")
                    try:
                        num = int(parts[1])
                    except ValueError:
                        num = None
                    if parts[0] == "cuda" and num != None:
                        if num < torch.cuda.device_count():
                            self.device = settings["device"]
                        else:
                            print(f"Invalid device identifier {settings['device']} with only {torch.cuda.device_count()} devices detected.")
                    else:
                        print(f"Invalid cuda device identifier {settings['device']}.")
                else:
                    print(f"Invalid device identifier {settings['device']}.")
            else:
                print(f"Field 'device' in settings needs to be a string.")

        if "audio_file" in settings:
            if isinstance(settings["audio_file"], str):
                if settings["audio_file"] == "all":
                    self.audio_file = settings["audio_file"]
                elif settings["audio_file"].endswith(".wav"):
                    file_path = os.path.join("in", settings["audio_file"])
                    if os.path.exists(file_path):
                        self.audio_file = settings["audio_file"]
                    else:
                        print(f"Audio file {settings['audio_file']} specified in settings does not exist.")
                else:
                    print(f"Invalid audio file {settings['audio_file']} specified in settings.")
            else:
                print(f"Field 'audio_file' in settings needs to be a string.")

    def model_selection_loop(self, dir: str):
        while self.model_path == "" or self.index_file == "":
            dir_contents = os.listdir(dir)
            dir_contents.sort()
            models = [path for path in dir_contents if os.path.isfile(os.path.join(dir, path))]
            categories = [path for path in dir_contents if os.path.isdir(os.path.join(dir, path)) and path != "index"]
            if len(models) == 0 and len(categories) == 0:
                print(f"There are no models inside the directory {dir}")
                sys.exit(1)
            elif len(models) == 0:
                if dir != "models":
                    print(f"Location: {dir}")
                print("Select a category.")
            elif len(categories) == 0:
                if dir != "models":
                    (f"Location: {dir}")
                print("Select a model.")
            else:
                if dir != "models":
                    print(f"Location: {dir}")
                print("Select a model or a category.")
            for i in range(len(models)):
                print(f"[{i}] {models[i]}")
            print()
            for i in range(len(categories)):
                print(f"[{i+len(models)}] {categories[i]}")

            model_selection = input("> ")
            try:
                selection_num = int(model_selection)
                if 0 <= selection_num <= len(models) + len(categories) - 1:
                    if selection_num < len(models):
                        self.model_path = os.path.join(dir, models[selection_num])
                        model_name = models[selection_num].split(".")[0]
                        matching_index_files = []
                        for file in os.listdir(os.path.join("models", "index")):
                            if os.path.isfile(os.path.join("models", "index", file)):
                                if file.endswith(f"{model_name}_v1.index") or file.endswith(f"{model_name}_v2.index") or file.endswith(f"{model_name}.index"):
                                    matching_index_files.append(file)
                        if len(matching_index_files) == 1:
                            self.index_file = os.path.join("models", "index", matching_index_files[0])
                        elif len(matching_index_files) == 0:
                            print(f"The index for the model {model_name} could not be determined.")
                            sys.exit(1)
                        else:
                            print(f"Found multiple matching indexes {matching_index_files} for model {model_name}.")
                            sys.exit(1)
                    else:
                        self.model_selection_loop(os.path.join(dir, categories[selection_num-len(models)]))
                else:
                    print("Invalid selection. Try again.")
            except ValueError:
                print("Invalid selection. Try again.")

    def fill_remaining_params(self):
        # select device
        while self.device == "":
            print("Select a device to perform inference on.")
            print("[0] cpu")
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                print(f"[{i+1}] cuda:{i} \"{name}\"")
            device_selection = input("> ")
            try:
                selection_num = int(device_selection)
                if selection_num == 0:
                    self.device = "cpu"
                elif selection_num > torch.cuda.device_count() or selection_num < 0:
                    print("Invalid selection. Try again.")
                else:
                    self.device = f"cuda:{selection_num-1}"
            except ValueError:
                print("Invalid selection. Try again.")

        # select model
        self.model_selection_loop("models")

        # select pitch adjustment
        while self.pitch_adjustment == None:
            print("Enter the number of semitones to adjust the audio by.")
            print("Lowest value: -12 Lower by an octave.")
            print("Highest value: 12 Raise by an octave.")
            pitch_input = input("> ")
            try:
                pitch = int(pitch_input)
                if -12 <= pitch <= 12:
                    self.pitch_adjustment = pitch
                else:
                    print("Invalid selection. Try again.")
            except ValueError:
                print("Invalid selection. Try again.")


        # select pitch extraction method
        while self.pitch_extraction_method == "":
            print("Select a method for pitch extraction.")
            print(f"[0] pm: faster extraction but lower-quality speech")
            print(f"[1] harvest: better bass but extremely slow")
            print(f"[2] crepe: better quality but GPU intensive")
            print(f"[3] rmvpe: best quality and little GPU requirement")
            options = ["pm", "harvest", "crepe", "rmvpe"]
            method_selection = input("> ")
            try:
                selection_num = int(method_selection)
                if 0 <= selection_num <= len(options):
                    self.pitch_extraction_method = options[selection_num]
                else:
                    print("Invalid selection. Try again.")
            except ValueError:
                print("Invalid selection. Try again.")

        # select audio
        if self.audio_file == "":
            loop = True
            while loop:
                print("Select the audio file(s) to use. Separate multiple selections with ';'")
                audio_files = [file for file in os.listdir("in") if file.endswith(".wav")]
                if len(audio_files) == 0:
                    print("No audio files available.")
                    sys.exit(1)
                for i in range(len(audio_files)):
                    print(f"[{i}] {audio_files[i]}")
                print(f"[{len(audio_files)}] All Files")
                audio_selection = input("> ")
                try:
                    selection_num = int(audio_selection)
                    if 0 <= selection_num < len(audio_files):
                        input_file = os.path.join("in", audio_files[selection_num])
                        output_file = os.path.join("out", audio_files[selection_num])
                        infer(self.pitch_adjustment, input_file, output_file, self.model_path, self.index_file, self.device, self.pitch_extraction_method)
                        loop = False
                    elif selection_num == len(audio_files):
                        for i in range(audio_files):
                            input_file = os.path.join("in", audio_files[i])
                            output_file = os.path.join("out", audio_files[i])
                            infer(self.pitch_adjustment, input_file, output_file, self.model_path, self.index_file, self.device, self.pitch_extraction_method)
                        loop = False
                    else:
                        print("Invalid selection. Try again.")
                except ValueError:
                    print("Invalid selection. Try again.")
        elif self.audio_file == "all":
            audio_files = [file for file in os.listdir("in") if file.endswith(".wav")]
            for i in range(audio_files):
                input_file = os.path.join("in", audio_files[i])
                output_file = os.path.join("out", audio_files[i])
                infer(self.pitch_adjustment, input_file, output_file, self.model_path, self.index_file, self.device, self.pitch_extraction_method)
        else:
            input_file = os.path.join("in", self.audio_file)
            output_file = os.path.join("out", self.audio_file)
            infer(self.pitch_adjustment, input_file, output_file, self.model_path, self.index_file, self.device, self.pitch_extraction_method)


def folder_check():
    for folder in ["in", "out", "models", os.path.join("models", "index")]:
        if not os.path.exists(folder):
            os.mkdir(folder)
    if not os.path.exists("settings.json"):
        with open("settings.json", "w") as file:
            json.dump({}, file, indent=4)
    for file in ["rvc/hubert_base.pt", "rvc/rmvpe.pt"]:
        if not os.path.exists(file):
            print(f"Model file {file} does not exist. Please download them. A possible source is listed in the readme.")
            sys.exit(1)


if __name__ == "__main__":
    folder_check()
    interface = CLI_Interface()
    interface.load_settings()
    interface.fill_remaining_params()
