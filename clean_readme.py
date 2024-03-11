import re

def center_wrap_img_tags_in_file(readme_path, output_path=None):
    # If no output path is provided, overwrite the original file
    if output_path is None:
        output_path = readme_path

    with open(readme_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # This pattern matches all <img> tags
    pattern = r'(<img.+?>)'
    replacement = r'<p align="center">\1</p>'
    
    # Replace all <img> tags with centered versions
    modified_content = re.sub(pattern, replacement, content)

    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(modified_content)

# Example usage:
readme_path = 'README.md'
# If you want the changes in the same README.md file, call the function with only one argument
center_wrap_img_tags_in_file(readme_path)
