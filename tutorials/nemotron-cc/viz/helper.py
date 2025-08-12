import logging

import matplotlib.pyplot as plt


class DataSizeTracker:
    def __init__(self, original_size: int) -> None:
        self.original_size = original_size
        self.data = {}
        self.data_list = []  # Store data in a list to track order
        self.logger = logging.getLogger(__name__)

    def record_size(self, stage_name: str, size: int = 0) -> None:
        if size == -1:
            size = self.data_list[-1][1] if self.data_list else self.original_size
        self.data[stage_name] = size
        self.data_list.append((stage_name, size))  # Append to the list
        self.logger.debug(f"Recorded size for {stage_name}: {size}")

    def get_data(self) -> dict:
        return self.data

    def calculate_incremental_change(self, stage_name: str) -> float | None:
        if stage_name not in self.data:
            self.logger.warning(f"Stage {stage_name} not found in data.")
            return None

        # Find the current stage's index and get its size
        current_index = next((i for i, (name, _) in enumerate(self.data_list) if name == stage_name), -1)
        current_size = self.data[stage_name]

        # Get previous stage's size, or use original size if this is the first stage
        previous_size = self.original_size
        if current_index > 0:
            previous_size = self.data_list[current_index - 1][1]

        filtered_out = previous_size - current_size

        if filtered_out != 0:  # noqa: SIM108
            # Calculate percentage based on original size
            percent_filtered = (filtered_out / self.original_size) * 100
        else:
            percent_filtered = 0.0

        self.logger.debug(f"Incremental change for {stage_name}: {filtered_out}")
        return filtered_out, percent_filtered

    def calculate_overall_change(self) -> float:
        if not self.data:
            self.logger.warning("No data recorded yet.")
            return 0

        final_stage = self.data_list[-1][0]
        final_size = self.data[final_stage]
        total_filtered = self.original_size - final_size

        if self.original_size != 0:  # noqa: SIM108
            # Calculate what percentage of original data was filtered out
            percent_filtered = (total_filtered / self.original_size) * 100
        else:
            percent_filtered = 0.0

        self.logger.debug(f"Overall filtered amount: {total_filtered}")
        return total_filtered, percent_filtered

    def print_summary(self) -> None:
        print("Data Processing Summary:")
        print(f"Original Size: {self.original_size}")
        print(f"Final Size: {self.data_list[-1][1]}")
        total_filtered, total_percent = self.calculate_overall_change()
        print(f"Total Filtered: {total_filtered} ({total_percent:.2f}% of original)")

        print("\nStage-wise Filtering:")
        for stage, size in self.data.items():
            filtered_amount, percent = self.calculate_incremental_change(stage)
            if filtered_amount is not None:
                print(
                    f"  Stage: {stage}, Result Size: {size}, Filtered Out: {filtered_amount} ({percent:.2f}% of original)"
                )
            else:
                print(f"  Stage: {stage}, Size: {size}, Filtered Amount: Not Available")

    def plot_size_reduction(self) -> None:
        """Plot the dataset size reduction after each filtering stage."""
        stages = ["Original"] + [stage for stage, _ in self.data_list]
        sizes = [self.original_size] + [size for _, size in self.data_list]

        # Calculate percentages of original size
        percentages = [(size / self.original_size) * 100 for size in sizes]

        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        ax.plot(range(len(stages)), percentages, marker="o")

        # Rotate x-axis labels for better readability
        ax.set_xticks(range(len(stages)))
        ax.set_xticklabels(stages, rotation=45, ha="right")

        ax.set_title("Dataset Size Reduction by Filtering Stage")
        ax.set_xlabel("Filtering Stage")
        ax.set_ylabel("Percentage of Original Size")

        # Add percentage labels on top of each point
        for i, pct in enumerate(percentages):
            label = f"{pct:.1f}%"
            ax.text(i, pct, label, ha="center", va="bottom")

        ax.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()  # Adjust layout to prevent label cutoff
        plt.show()
