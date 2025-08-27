def resolve_filename_column(add_filename_column: bool | str) -> str | None:
    """Resolve the filename column name based on the input parameter.

    Args:
        add_filename_column: Can be:
            - True: Use default filename column name
            - False: No filename column
            - str: Use the provided string as column name

    Returns:
        str | None: The filename column name or None if not needed
    """
    if add_filename_column is True:
        return "file_name"
    elif add_filename_column is False:
        return None
    elif isinstance(add_filename_column, str):
        return add_filename_column
    else:
        msg = f"Invalid value for add_filename_column: {add_filename_column}"
        raise ValueError(msg)
