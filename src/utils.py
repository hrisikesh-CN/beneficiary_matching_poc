def dict_to_markdown_table(data_dict):
    """
    Convert a dictionary into a Markdown table.
    
    Parameters:
    data_dict (dict): The dictionary to convert, with keys as column headers.
    table_name (str): The title of the table in Markdown (optional).
    
    Returns:
    str: A string containing the Markdown table.
    """
    # Create the header
    markdown_table = "| K | Silhoutte Score |\n"
    markdown_table += "| ----- | ----- |\n"
    
    # Add rows from the dictionary
    for key, value in data_dict.items():
        markdown_table += f"| {key} | {value} |\n"
    
    return markdown_table