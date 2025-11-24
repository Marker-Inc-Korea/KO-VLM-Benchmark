class MissingColumnError(Exception):
    def __init__(self, missing_columns: list[str]):
        message = f"Missing required columns: {', '.join(missing_columns)}"
        super().__init__(message)
