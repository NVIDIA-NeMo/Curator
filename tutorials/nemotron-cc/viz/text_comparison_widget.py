import uuid
from typing import Any

import pandas as pd
from IPython.display import HTML


def text_comparison_widget(  # noqa: PLR0913
    df: pd.DataFrame,
    id_column: str = "id",
    col1: str = "original_text",
    col2: str = "processed_text",
    title1: str | None = None,
    title2: str | None = None,
    max_height: str = "400px",
    width: str = "100%",
) -> HTML:
    """
    Create a beautiful, scrollable side-by-side comparison of two text columns from a DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data
    id_column : str, default='id'
        The name of the column to use as the identifier
    col1 : str, default='original_text'
        The name of the first column to compare
    col2 : str, default='processed_text'
        The name of the second column to compare
    title1 : str, optional
        Custom title for the first column (defaults to col1 if None)
    title2 : str, optional
        Custom title for the second column (defaults to col2 if None)
    max_height : str, default='400px'
        The maximum height of each text cell
    width : str, default='100%'
        The width of the comparison table

    Returns:
    --------
    IPython.display.HTML
        HTML display object with the comparison
    """
    # Set default column titles if not provided
    title1 = title1 or col1.replace("_", " ").title()
    title2 = title2 or col2.replace("_", " ").title()

    # Generate a unique ID for this instance
    widget_id = f"text-comparison-{str(uuid.uuid4())[:8]}"

    # Define theme colors
    themes = {
        "nvidia": {
            "bg": "#202020",
            "header_bg": "#151515",
            "border": "#538300",
            "id_bg": "#202020",
            "text_bg": "#282828",
            "text_color": "#E0E0E0",
            "header_text": "#76B900",
            "hover": "#303030",
            "button_bg": "#538300",
            "button_hover": "#76B900",
            "button_text": "#FFFFFF",
            "scrollbar_track": "#252525",
            "scrollbar_thumb": "#538300",
        }
    }

    # Use the selected theme (default to dark if invalid)
    colors = themes["nvidia"]

    # Create HTML for the comparison
    html_code = f"""
    <style>
        /* Main container */
        #{widget_id}-container {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: {colors["bg"]};
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            margin: 20px 0;
            width: {width};
            border: 1px solid {colors["border"]};
        }}

        /* Table styling */
        #{widget_id} {{
            width: 100%;
            border-collapse: collapse;
            border-spacing: 0;
            background-color: {colors["bg"]};
        }}

        /* Header styling */
        #{widget_id} th {{
            background-color: {colors["header_bg"]};
            color: {colors["header_text"]};
            font-weight: 600;
            font-size: 14px;
            text-align: left;
            padding: 16px;
            border-bottom: 2px solid {colors["border"]};
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        /* Cell styling */
        #{widget_id} td {{
            padding: 0;
            border-bottom: 1px solid {colors["border"]};
            vertical-align: top;
        }}

        /* ID column */
        #{widget_id} .row-id {{
            font-weight: 600;
            padding: 16px;
            background-color: {colors["id_bg"]};
            color: {colors["text_color"]};
            font-size: 13px;
            text-align: center;
            width: 100px;
            max-width: 100px;
            overflow: hidden;
            text-overflow: ellipsis;
            box-shadow: 1px 0 0 {colors["border"]};
        }}

        /* Text cells */
        #{widget_id} .text-cell {{
            overflow-y: auto;
            max-height: {max_height};
            padding: 16px;
            white-space: pre-wrap;
            word-wrap: break-word;
            background-color: {colors["text_bg"]};
            color: {colors["text_color"]};
            font-size: 14px;
            line-height: 1.6;
            transition: background-color 0.2s;
        }}

        /* Hover effects */
        #{widget_id} tr:hover .text-cell {{
            background-color: {colors["hover"]};
        }}

        #{widget_id} tr:hover .row-id {{
            background-color: {colors["hover"]};
        }}

        /* Controls styling */
        #{widget_id}-controls {{
            padding: 12px 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: {colors["header_bg"]};
            border-bottom: 1px solid {colors["border"]};
        }}

        #{widget_id}-title {{
            font-weight: 600;
            font-size: 16px;
            color: {colors["header_text"]};
        }}

        #{widget_id}-buttons {{
            display: flex;
            gap: 8px;
        }}

        #{widget_id}-buttons button {{
            padding: 8px 12px;
            border-radius: 6px;
            border: none;
            background-color: {colors["button_bg"]};
            color: {colors["button_text"]};
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        }}

        #{widget_id}-buttons button:hover {{
            background-color: {colors["button_hover"]};
        }}

        /* Custom scrollbar styling */
        #{widget_id} .text-cell::-webkit-scrollbar {{
            width: 8px;
        }}

        #{widget_id} .text-cell::-webkit-scrollbar-track {{
            background: {colors["scrollbar_track"]};
            border-radius: 4px;
        }}

        #{widget_id} .text-cell::-webkit-scrollbar-thumb {{
            background: {colors["scrollbar_thumb"]};
            border-radius: 4px;
        }}

        #{widget_id} .text-cell::-webkit-scrollbar-thumb:hover {{
            background: {colors["button_hover"]};
        }}

        /* Difference highlight */
        #{widget_id} .diff-highlight {{
            background-color: rgba(255, 230, 0, 0.3);
            border-radius: 2px;
        }}

        /* Footer */
        #{widget_id}-footer {{
            font-size: 12px;
            color: {colors["header_text"]};
            padding: 8px 16px;
            text-align: right;
            background-color: {colors["header_bg"]};
            border-top: 1px solid {colors["border"]};
        }}

        /* Responsive adjustments */
        @media (max-width: 768px) {{
            #{widget_id} .row-id {{
                width: 60px;
                max-width: 60px;
                font-size: 12px;
                padding: 12px 8px;
            }}

            #{widget_id} .text-cell {{
                padding: 12px;
                font-size: 13px;
            }}

            #{widget_id} th {{
                padding: 12px;
                font-size: 13px;
            }}
        }}
    </style>

    <div id="{widget_id}-container">
        <div id="{widget_id}-controls">
            <div id="{widget_id}-title">Text Comparison</div>
            <div id="{widget_id}-buttons">
                <button onclick="toggleWordWrap('{widget_id}')">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <polyline points="17 1 21 5 17 9"></polyline>
                        <path d="M3 11V9a4 4 0 0 1 4-4h14"></path>
                        <polyline points="7 23 3 19 7 15"></polyline>
                        <path d="M21 13v2a4 4 0 0 1-4 4H3"></path>
                    </svg>
                    <span style="margin-left: 5px;">Word Wrap</span>
                </button>
                <button onclick="increaseFontSize('{widget_id}')">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="12" cy="12" r="10"></circle>
                        <line x1="12" y1="8" x2="12" y2="16"></line>
                        <line x1="8" y1="12" x2="16" y2="12"></line>
                    </svg>
                    <span style="margin-left: 5px;">Larger</span>
                </button>
                <button onclick="decreaseFontSize('{widget_id}')">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="12" cy="12" r="10"></circle>
                        <line x1="8" y1="12" x2="16" y2="12"></line>
                    </svg>
                    <span style="margin-left: 5px;">Smaller</span>
                </button>
            </div>
        </div>

        <table id="{widget_id}">
            <thead>
                <tr>
                    <th>{id_column.replace("_", " ").title()}</th>
                    <th>{title1}</th>
                    <th>{title2}</th>
                </tr>
            </thead>
            <tbody>
    """

    # Add each row to the HTML
    for _, row in df.iterrows():
        row_id = row[id_column]
        text1 = str(row[col1]) if pd.notna(row[col1]) else ""
        text2 = str(row[col2]) if pd.notna(row[col2]) else ""

        html_code += f"""
        <tr>
            <td class="row-id">{row_id}</td>
            <td><div class="text-cell">{text1}</div></td>
            <td><div class="text-cell">{text2}</div></td>
        </tr>
        """

    html_code += """
            </tbody>
        </table>
        <div id="{widget_id}-footer">
            Use mouse wheel to scroll each cell independently
        </div>
    </div>

    <script>
        function toggleWordWrap(widgetId) {
            const cells = document.querySelectorAll(`#${widgetId} .text-cell`);
            cells.forEach(cell => {
                if (cell.style.whiteSpace === 'nowrap') {
                    cell.style.whiteSpace = 'pre-wrap';
                } else {
                    cell.style.whiteSpace = 'nowrap';
                }
            });
        }

        function increaseFontSize(widgetId) {
            const cells = document.querySelectorAll(`#${widgetId} .text-cell`);
            cells.forEach(cell => {
                const currentSize = window.getComputedStyle(cell).fontSize;
                const newSize = parseFloat(currentSize) + 1;
                cell.style.fontSize = `${newSize}px`;
            });
        }

        function decreaseFontSize(widgetId) {
            const cells = document.querySelectorAll(`#${widgetId} .text-cell`);
            cells.forEach(cell => {
                const currentSize = window.getComputedStyle(cell).fontSize;
                const newSize = Math.max(10, parseFloat(currentSize) - 1);
                cell.style.fontSize = `${newSize}px`;
            });
        }
    </script>
    """

    return HTML(html_code.replace("{widget_id}", widget_id))


def compare_row_by_id(  # noqa: PLR0913
    df: pd.DataFrame,
    row_id: Any,  # noqa: ANN401
    id_column: str = "id",
    col1: str = "original_text",
    col2: str = "processed_text",
    title1: str | None = None,
    title2: str | None = None,
    max_height: str = "400px",
    width: str = "100%",
) -> HTML:
    """
    Create a scrollable side-by-side comparison of two text columns for a specific row ID.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data
    row_id : any
        The identifier value to filter for
    id_column : str, default='id'
        The name of the column to use as the identifier
    col1 : str, default='original_text'
        The name of the first column to compare
    col2 : str, default='processed_text'
        The name of the second column to compare
    title1 : str, optional
        Custom title for the first column (defaults to col1 if None)
    title2 : str, optional
        Custom title for the second column (defaults to col2 if None)
    max_height : str, default='400px'
        The maximum height of each text cell
    width : str, default='100%'
        The width of the comparison table
    Returns:
    --------
    IPython.display.HTML
        HTML display object with the comparison
    """
    # Filter for the specific row
    filtered_df = df[df[id_column] == row_id]

    if len(filtered_df) == 0:
        return HTML(
            f"<div style=\"padding: 16px; background-color: #fee2e2; color: #b91c1c; border-radius: 8px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;\">No row found with {id_column} = {row_id}</div>"
        )

    return text_comparison_widget(filtered_df, id_column, col1, col2, title1, title2, max_height, width)
