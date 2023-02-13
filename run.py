import timeit
import torch
import os
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM

class ModelController:
    models = {
        'sber-large': 'sberbank-ai/rugpt3large_based_on_gpt2',
        'sber-small': 'sberbank-ai/rugpt3small_based_on_gpt2',
        'anecdote': 'repositories/nlp-course/anecdote'
    }
    def __init__(self, model_name: str) -> None:
        self.my_device = torch.device("cpu")
        self.current_model = self.models[model_name]
        self.model = AutoModelForCausalLM.from_pretrained(self.current_model).to(self.my_device)
        if model_name == 'anecdote':
            self.tokenizer = AutoTokenizer.from_pretrained('sberbank-ai/rugpt3large_based_on_gpt2')
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.current_model)

        self.pipeline = pipeline(
            task='text-generation',
            tokenizer=self.tokenizer,
            model=self.model,
            device=self.my_device
        )
    
    def generate_text(self, start_str, max_length, repetition_penalty, temperature, number_of_seq, do_sample=True):
        return self.pipeline(
            start_str, 
            max_length=max_length,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            temperature=temperature,
            num_return_sequences=number_of_seq
            )

model = ModelController('anecdote')

def show_time_difference(func):
    def _wrapper(*args, **kwargs):
        starttime = timeit.default_timer()
        print("The start time is:", starttime)
        func(*args, **kwargs)
        print("The time difference is:", timeit.default_timer() - starttime)
    return _wrapper

@show_time_difference
def batch_text_generation_to_files(
                            text: str, 
                            folder: str, 
                            files_count: int, 
                            model: ModelController, 
                            max_length: int, 
                            repetition_penalty: float, 
                            temperature: float) -> None:
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/batch_generation/'
    path = ROOT_DIR + folder
    try:
        os.mkdir(path)
    except OSError as error:
        print(error)
        exit

    for i in range(1, files_count + 1):
        with open(f'{path}/{i}.txt', 'w+') as f:
            print(
                model.generate_text(
                    start_str=text,
                    max_length=max_length,
                    repetition_penalty=repetition_penalty,
                    do_sample=True,
                    temperature=temperature,
                ),
                file=f
            )

@show_time_difference
def batch_text_generation_to_stdout(
                            text: str, 
                            sequency_length: int, 
                            model: ModelController, 
                            max_length: int, 
                            repetition_penalty: float, 
                            temperature: float,
                            do_sample: bool) -> None:
    for i in range(0, sequency_length):
        print( 
            model.generate_text(
                start_str=text,
                max_length=max_length,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                do_sample=do_sample,
                number_of_seq=seq_number
            )[i]['generated_text']
        )
        print('\n\n________________________________________\n\n')


# batch_text_generation_to_files(
#     text='Упал однажды Лёня с трубы,',
#     folder='1',
#     files_count=10,
#     model=model,
#     max_length=150,
#     repetition_penalty=1.2,
#     temperature=1.0
# )

seq_number = 3

batch_text_generation_to_stdout(
    text='Продавец предложил мне',
    sequency_length=3,
    model=model,
    max_length=150,
    repetition_penalty=1.45,
    temperature=0.9,
    do_sample=True
)
