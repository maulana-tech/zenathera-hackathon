# Implementation Plan - Streamlit UI/UX Improvements

The goal is to enhance the user experience of the Streamlit dashboard by improving the visual theme and integrating missing visualizations.

## User Review Required
> [!NOTE]
> I will be updating the CSS to a more modern, "premium" look. This involves changing colors, shadows, and spacing.

## Proposed Changes

### UI/UX Improvements
#### [MODIFY] [style.css](file:///Users/em/web/scarping-hackathon/streamlit_app/style.css)
- Update color palette to a more refined set (e.g., slate/indigo).
- make a dark and light mode theme
- Update font to a more modern, premium font (e.g., Inter).
- Update icon to a more modern, premium icon (e.g., Heroicons).
- Update button to a more modern, premium button (e.g., Tailwind).
- Update card to a more modern, premium card (e.g., Tailwind).
- Update table to a more modern, premium table (e.g., Tailwind).
- Add hover effects to cards.
- Improve sidebar styling.
- Custom styling for Streamlit tabs and headers.
- Add a subtle background gradient.

### Visualization Updates
#### [MODIFY] [pages/visualizations.py](file:///Users/em/web/scarping-hackathon/streamlit_app/pages/visualizations.py)
- Add a new tab for "Barchart" (Topik per Dokumen/Skor).
- add a `heatmap.html` file to the `assets` folder.
- add a `hierarchy.html` file to the `assets` folder.
- add a `document.html` file to the `assets` folder.
- add a `topics.html` file to the `assets` folder.
- Load `barchart.html` using `load_html_asset`.
- Add descriptive text for the new visualization.
- Improve layout consistency.

### Overview Page Enhancements
#### [MODIFY] [pages/overview.py](file:///Users/em/web/scarping-hackathon/streamlit_app/pages/overview.py)
- Ensure custom HTML cards align with the new CSS.
- Slightly adjust layout for better spacing.

## Verification Plan
### Manual Verification
- Run `streamlit run streamlit_app/main.py`.
- Check the "Overview" page for the new look and feel.
- Navigate to "Visualizations" and verify all 5 tabs (Topik, Dokumen, Hierarki, Heatmap, Barchart) load correctly.
- Verify that the `barchart.html` asset is displayed.
- Check responsiveness and general aesthetics.
