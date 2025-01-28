import re
from typing import Dict, List
from src.config.logging import setup_logger

logger = setup_logger(__name__)


def models_to_text(data: List[Dict]) -> List[Dict]:
    """Preprocess the data by combining relevant fields into a 'content' field."""
    logger.debug("Preprocessing data...")


    def format_model_description(raw_text):
        """
        Formats the raw model description by removing commas, structuring sentences, 
        and improving readability for semantic search.
        """
        # Remove curly braces, brackets, and underscores
        cleaned_text = re.sub(r"[\{\}\[\]]", "", raw_text)
        cleaned_text = re.sub(r"_", " ", cleaned_text)

        # Replace colons for sentence structuring
        cleaned_text = re.sub(r"\s*:\s*", ": ", cleaned_text)

        # Replace commas with periods for sentence separation
        cleaned_text = re.sub(r"\s*,\s*", ". ", cleaned_text)

        # Normalize spaces and capitalize each sentence
        sentences = re.split(r"(?<=\.)\s*", cleaned_text.strip())
        formatted_text = ". ".join(sentence.capitalize() for sentence in sentences)

        return formatted_text

    def dict_to_string(data):
        try:
            # Recursive function to handle nested dictionaries and lists
            def convert_value(value):
                if isinstance(value, dict):
                    # Handle special MongoDB-style number representations
                    if "$numberDouble" in value:
                        return str(value["$numberDouble"])
                    elif "$numberInt" in value:
                        return str(value["$numberInt"])
                    return str(value)
                elif isinstance(value, list):
                    return ", ".join(map(str, value))
                return str(value)

            # Flatten the dictionary into a single string
            string_parts = []
            for key, value in data.items():
                # Handle nested dictionaries
                if isinstance(value, dict):
                    # Special handling for nested metrics or complex nested structures
                    if key in [
                        "performance",
                        "hardware_requirements",
                        "popularity",
                    ]:
                        nested_parts = [
                            f"{sub_key}: {convert_value(sub_value)}"
                            for sub_key, sub_value in value.items()
                        ]
                        string_parts.append(f"{key}: {{{', '.join(nested_parts)}}}")
                    else:
                        string_parts.append(f"{key}: {convert_value(value)}")
                else:
                    string_parts.append(f"{key}: {convert_value(value)}")

            return ", ".join(string_parts)

        except Exception:
            return (
                f"{data.get('name', '')} "
                f"{data.get('framework', '')} "
                f"{', '.join(data.get('task', []))} "  # Joining 'task' list into a string
                f"{data.get('architecture', '')} "
                f"{', '.join(data.get('domains', []))} "  # Joining 'domains' list into a string
                f"{', '.join(data.get('use_cases', []))} "  # Joining 'use_cases' list into a string
                f"{data.get('license', '')} "
                f"{data.get('popularity', {}).get('stars', '')} "
                f"{data.get('popularity', {}).get('downloads', '')} "
                f"{data.get('model_size_parameters', {})} "
                f"{data.get('hardware_requirements', {})} "
            )

    for item in data:
        item["content"] = format_model_description(dict_to_string(item))

    logger.debug("Data preprocessing completed.")
    return data