import yaml
import sys
sys.path.append('scripts')
from config_generator import generate_config

y = generate_config('test', 'model', 'data', 'out', 64, 64, 6, 1e-4, 100, 1, 1024, True, 64, 512, 1024, False, False, 'adam', 'const', 10, True, False, 100, 100, ['A prompt with "quotes"', 'And : colons', '[brackets]'])
parsed = yaml.safe_load(y)
print(parsed['config']['process'][0]['datasets'][0]['folder_path'])
print(isinstance(parsed['config']['process'][0]['train']['sample_prompts'], list))
print(parsed['config']['process'][0]['model']['quantize'])
