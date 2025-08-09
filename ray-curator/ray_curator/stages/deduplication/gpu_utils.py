def align_memory(memory_size: int) -> int:
    """
    Aligns a memory size to the nearest multiple of 256.
    """
    return (memory_size // 256) * 256
