import os
import jinja2
import yaml
import datetime

template_dir = '/ncrc/home2/Tsung-Lin.Hsieh/nudge-to-fine-stellar-workflow/configs/templates'
template_files = [
    'training-data-tendencies.yaml',
    'training-data-fluxes.yaml',
    'testing-data-tendencies.yaml',
    'testing-data-fluxes.yaml',
    'validation-data-tendencies.yaml',
    'validation-data-fluxes.yaml'
    ]

for template_file in template_files:
    
    ## fill the template
    template_loader = jinja2.FileSystemLoader(searchpath=template_dir)
    template_env = jinja2.Environment(loader=template_loader, autoescape=True)
    template = template_env.get_template(template_file)

    result = template.render(
        CATALOG='_CATALOG_',
        SIMULATION_ROOT='_SIMULATION_ROOT_',
        RADIATIVE_FLUX_DATASET='_RADIATIVE_FLUX_DATASET_'
    )

    with open(f'tmp_filled_template.yaml', 'w') as file:
        file.write(result)

    ## load the filled template
    with open(f'tmp_filled_template.yaml', 'r') as file:
        data = yaml.safe_load(file)

    ## add 10 days to the timesteps
    delta = datetime.timedelta(days=10) #i

    datetime_format = "%Y%m%d.%H%M%S"
    for i in range(len(data['timesteps'])):
        data['timesteps'][i] = (datetime.datetime.strptime(data['timesteps'][i], datetime_format) + delta).strftime(datetime_format)

    ## write the modified timesteps to a new file
    with open(f'tmp_modified_template.yaml', 'w') as file:
        yaml.dump(data, file)

    ## replace text
    with open(f'tmp_modified_template.yaml', 'r') as file:
        data = file.read()
        data = data.replace('_CATALOG_', '{{ CATALOG }}')
        data = data.replace('_SIMULATION_ROOT_', '{{ SIMULATION_ROOT }}')
        data = data.replace('_RADIATIVE_FLUX_DATASET_', '{{ RADIATIVE_FLUX_DATASET }}')

    with open(f'tmp_modified_template.yaml', 'w') as file:
        file.write(data)

    ## move the modified file to the original directory
    os.system(f'mv tmp_modified_template.yaml {template_dir}/{template_file[:-5]}_0129.yaml')

    os.system(f'rm tmp_filled_template.yaml')
