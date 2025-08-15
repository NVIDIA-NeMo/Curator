def align_down_to_256(memory_size: int) -> int:
    """
    Aligns a memory size down to the nearest multiple of 256.
    """
    return (memory_size // 256) * 256
