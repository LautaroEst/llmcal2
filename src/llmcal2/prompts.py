

class Llama3Prompt:

    def __init__(self, max_characters=400):
        self.max_characters = max_characters
        self.prompt = None

    def apply(self, text):
        filled_prompts = []
        for t in text:
            filled_prompts.append(self.prompt.replace("{inpt}", t[:self.max_characters]))
        return filled_prompts
    
    def fit(self, prompt_template, shots):
        preface = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{prompt_template}<|eot_id|>" # No newline
        )
        output_preface = (
            "<|start_header_id|>user<|end_header_id|>\n\n{inpt}<|eot_id|>" # No newline
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        
        if len(shots) == 0:
            return preface + output_preface
        
        shot_template = (
            "<|start_header_id|>user<|end_header_id|>\n\n{shot_inpt}<|eot_id|>" # No newline
            "<|start_header_id|>assistant<|end_header_id|>\n\n{shot_label}<|eot_id|>" # No newline
        )
        shots_prompt = ""
        for shot in shots:
            shots_prompt += shot_template.format(shot_inpt=shot["text"][:self.max_characters], shot_label=shot["label"])
        self.prompt = preface + shots_prompt + output_preface

        return self
    

class TinyLlamaPrompt:

    def __init__(self, max_characters=400):
        self.max_characters = max_characters
        self.prompt = None

    def apply(self, text):
        filled_prompts = []
        for t in text:
            filled_prompts.append(self.prompt.format(inpt=t[:self.max_characters]))
        return filled_prompts
    
    def fit(self, prompt_template, shots):
        preface = (
            "<|system|>\n"
            f"{prompt_template}</s>\n"
        )
        output_preface = (
            "<|user|>\n{inpt}</s>\n"
            "<|assistant|>\n"
        )
        
        if len(shots) == 0:
            return preface + output_preface
        
        shot_template = (
           "<|user|>\n{shot_inpt}</s>\n"
            "<|assistant|>\n{shot_label}</s>\n"
        )
        shots_prompt = ""
        for shot in shots:
            shots_prompt += shot_template.format(shot_inpt=shot["text"][:self.max_characters], shot_label=shot["label"])
        self.prompt = preface + shots_prompt + output_preface

        return self
    

class Phi3Prompt:

    def __init__(self, max_characters=400):
        self.max_characters = max_characters
        self.prompt = None

    def apply(self, text):
        filled_prompts = []
        for t in text:
            filled_prompts.append(self.prompt.format(inpt=t[:self.max_characters]))
        return filled_prompts
    
    def fit(self, prompt_template, shots):
        preface = (
            f'<|system|>\n{prompt_template}<|end|>\n'
        )
        output_preface = (
            "<|user|>\n{inpt}<|end|>\n"
            "<|assistant|>\n"
        )
        
        if len(shots) == 0:
            return preface + output_preface
        
        shot_template = (
            "<|user|>\n{shot_inpt}<|end|>\n"
            "<|assistant|>\n{shot_label}<|end|>\n"
        )
        shots_prompt = ""
        for shot in shots:
            shots_prompt += shot_template.format(shot_inpt=shot["text"][:self.max_characters], shot_label=shot["label"])
        self.prompt = preface + shots_prompt + output_preface

        return self
    

class PythiaPrompt:

    def __init__(self, max_characters=400):
        self.max_characters = max_characters
        self.prompt = None

    def apply(self, text):
        filled_prompts = []
        for t in text:
            filled_prompts.append(self.prompt.format(inpt=t[:self.max_characters]))
        return filled_prompts
    
    def fit(self, prompt_template, shots):
        preface = (
            f'<|system|>\n{prompt_template}<|end|>\n'
        )
        output_preface = (
            "<|user|>\n{inpt}<|end|>\n"
            "<|assistant|>\n"
        )
        
        if len(shots) == 0:
            return preface + output_preface
        
        shot_template = (
            "<|user|>\n{shot_inpt}<|end|>\n"
            "<|assistant|>\n{shot_label}<|end|>\n"
        )
        shots_prompt = ""
        for shot in shots:
            shots_prompt += shot_template.format(shot_inpt=shot["text"][:self.max_characters], shot_label=shot["label"])
        self.prompt = preface + shots_prompt + output_preface

        return self