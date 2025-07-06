import re
import random
import string
import os

def generate_random_hash(length=5):
    """Generate a random hash string of specified length"""
    return ''.join(random.choices(string.hexdigits, k=length)).upper()

def convert_curve_format(curve_data):
    """
    Convert curve data from lua format with indexed keys to lrtemplate format.
    
    Input format: [1] = 0, [2] = 19, ...
    Output format: 0, 19, ...
    """
    # Extract values using regex
    values = []
    pattern = r'\[\d+\]\s*=\s*(-?\d+(?:\.\d+)?)'
    matches = re.findall(pattern, curve_data)
    
    if not matches:
        return "{}"
    
    # Format as lrtemplate style
    formatted_values = []
    for value in matches:
        formatted_values.append(f"\t\t\t{value}")
    
    return "{\n" + ",\n".join(formatted_values) + ",\n\t\t}"
    
def process_lua_content(content):
    """Process lua content and extract/transform the required data"""
    # Find all curve data that needs conversion
    curve_keys = [
        'ToneCurvePV2012', 
        'ToneCurvePV2012Red', 
        'ToneCurvePV2012Green', 
        'ToneCurvePV2012Blue',
        'MainCurve',
        'RedCurve',
        'GreenCurve',
        'BlueCurve'
    ]
    
    # Extract each curve section and convert it
    for key in curve_keys:
        pattern = fr'{key}\s*=\s*\{{[^{{}}]*(\[\d+\][^{{}}]*)+\}}'
        match = re.search(pattern, content)
        if match:
            curve_section = match.group(0)
            curve_data = curve_section[curve_section.find('{')+1:curve_section.rfind('}')]
            converted_curve = convert_curve_format(curve_data)
            
            # Replace the original curve data with the converted format
            content = content.replace(curve_section, f"{key} = {converted_curve}")
    
    return content

def fix_indentation(content):
    """Fix the indentation of the content to match lrtemplate format"""
    # Split by lines
    lines = content.split('\n')
    indented_lines = []
    
    # Add proper indentation
    for line in lines:
        line = line.strip()
        if line:
            indented_lines.append(f"\t\t\t{line}")
    
    return '\n'.join(indented_lines)

def cleanup_content(content):
    """Clean up the content to remove extra curly braces and fix formatting"""
    # Remove the opening curly brace at the beginning if present
    content = re.sub(r'^\s*{\s*', '', content)
    
    # Remove the closing curly brace at the end if present
    content = re.sub(r'\s*}\s*$', '', content)
    
    return content

def lua_to_lrtemplate(lua_file_path, output_path=None):
    """Convert a Lightroom config.lua file to lrtemplate format"""
    if not output_path:
        # Generate output path based on input path if not provided
        base_name = os.path.splitext(os.path.basename(lua_file_path))[0]
        output_path = os.path.join(os.path.dirname(lua_file_path), f"{base_name}.lrtemplate")
    
    # Read the lua file
    with open(lua_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove the "return" at the beginning if present
    content = re.sub(r'^return\s*', '', content.strip())
    
    # Clean up the content
    content = cleanup_content(content)
    
    # Process the content to convert curve formats
    processed_content = process_lua_content(content)
    
    # Fix indentation for the entire content
    processed_content = fix_indentation(processed_content)
    
    # Generate random hash for internalName
    random_hash = generate_random_hash(5)
    preset_name = f"JarvisArt-{random_hash}"
    # Create the lrtemplate structure
    lrtemplate = f"""s = {{
	id = "",
	internalName = "Preset-{random_hash}",
	title = "{preset_name}",
	type = "Develop",
	value = {{
		settings = {{
{processed_content}
		}},
	}},
	version = 0,
}}
"""
    
    # Write the output file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(lrtemplate)
    
    return output_path

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python lua2lrt.py <input_lua_file> [output_lrtemplate_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = None
    
    # If a second argument is provided, use it as the output file path
    # Otherwise, use the same path as input but with .lrtemplate extension
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    try:
        output_path = lua_to_lrtemplate(input_file, output_file)
        print(f"Conversion successful! Output saved to: {output_path}")
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)
