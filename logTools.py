import os
import numpy as np
import uuid
from PIL import Image
from functools import wraps
import inspect
from imageOps import *

def is_iterable (x) : 
    try : 
        iter(x) 
        return True
    except Exception :
        return False

def log_to_dir(dir_name='logs'):
    """ Author: ChatGPT """ 
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create main directory if not exists
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            
            # Create a UUID for the subdirectory
            sub_dir = str(uuid.uuid4())
            sub_dir_path = os.path.join(dir_name, sub_dir)
            os.makedirs(sub_dir_path)
            
            # Log inputs
            metadata = []
            arg_names = inspect.getfullargspec(func).args

            for i, (name, arg) in enumerate(zip(arg_names, args)):
                if isinstance(arg, Image.Image):
                    arg.save(os.path.join(sub_dir_path, f'{name}.png'))
                elif isinstance(arg, np.ndarray): 
                    imgArrayToPIL(arg).save(os.path.join(sub_dir_path, f'{name}.png'))
                elif name == 'self' : 
                    continue
                else:
                    metadata.append(f'{name}: {arg}')
            
            for k, v in kwargs.items():
                if isinstance(v, Image.Image):
                    v.save(os.path.join(sub_dir_path, f'{k}.png'))
                else:
                    metadata.append(f'{k}: {v}')
            
            # Call the function
            result = func(*args, **kwargs)
            
            # See if the thing is an iterator
            if not is_iterable(result) : 
                result = (result,) 

            for i, res in enumerate(result):  
                # Log output
                if isinstance(res, Image.Image):
                    res.save(os.path.join(sub_dir_path, f'output_{i}.png'))
                elif isinstance(res, np.ndarray): 
                    imgArrayToPIL(res).save(os.path.join(sub_dir_path, f'output_{i}.png'))
                else:
                    metadata.append(f'output {i}: {res}')
            
            # Save metadata
            with open(os.path.join(sub_dir_path, 'metadata.txt'), 'w') as f:
                f.write('\n'.join(metadata))
            
            return result
        return wrapper
    return decorator

# Example usage
@log_to_dir()
def my_function(a, b):
    return a + b

@log_to_dir('custom_logs')
def another_function(img: Image.Image, text: str):
    return f"Received an image and text: {text}"

if __name__ == "__main__" :
    # Test the decorators
    my_function(1, 2)
    another_function(Image.new('RGB', (10, 10)), "Hello")



