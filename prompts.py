class PROMPTS:
    RATIO_ANALYSIS = """Analyze this keychain image and extract its dimensional proportions and characteristics.

**DIMENSIONAL ANALYSIS:**
Estimate the 3D proportions (Length : Width : Thickness):
- LENGTH: Longest horizontal dimension (use as baseline = 1.0)
- WIDTH: Longest vertical dimension (relative to length)
- THICKNESS: Depth from front to back surface

**THICKNESS ESTIMATION GUIDELINES:**
- Look at shadows, edges, and photo angle
- Flat/thin keychains (2-3mm): thickness = 0.05 - 0.1
- Standard keychains (3-5mm): thickness = 0.1 - 0.15
- Chunky/3D keychains (5-10mm): thickness = 0.2 - 0.4
- Very thick character figures: thickness = 0.4+


**IMPORTANT:** Ignore the keychain ring/chain hardware. Focus only on the main decorative piece.

Use the extract_keychain_proportions function to return structured data."""

    SILHOUETTE_EXTRACTION = """TECHNICAL SPECIFICATION for keychain shape extraction:

INPUT: Photograph of 3D-printed keychain
OUTPUT: 2D silhouette image (orthographic projection)

IMAGE PARAMETERS:
□ Dimensions: Square format (1:1 aspect ratio)
□ Resolution: 512x512px
□ Color depth: 1-bit (pure black & white, no grays)
□ Background: #FFFFFF (RGB: 255,255,255)
□ Foreground: #000000 (RGB: 0,0,0)
□ Format: PNG-8 or SVG-compatible raster


EXTRACTION RULES:
1. INCLUDE: Main decorative shape body + intentional voids/cutouts
2. EXCLUDE: Attachment hardware + background objects + lighting effects
3. PERSPECTIVE: Normalize to 0° viewing angle (front-facing)
4. SCALING: Object occupies 70-85% of canvas dimension
5. POSITIONING: Geometric center aligned to canvas center
6. EDGE QUALITY: Crisp boundaries, no feathering, no gradients

GEOMETRY PRESERVATION:
- Maintain aspect ratio from source
- Preserve symmetry axes if present
- Keep all holes/cutouts proportional
- Represent curves smoothly (not pixelated)
- Sharp corners remain sharp (no rounding)

VALIDATION CHECKLIST:
☐ Is the ring/chain removed?
☐ Is background completely white?
☐ Are all internal cutouts visible as white?
☐ Is the shape centered?
☐ Are edges sharp and clean?
☐ Is the viewing angle frontal?

Generate the extraction image meeting all specifications above."""

    SILHOUETTE_EDIT_PROMPT = """You are an expert graphic designer specializing in clean, vector-style silhouettes. Your task is to analyze a provided silhouette image with red markup and generate a revised, **pure black and white** version that incorporates all requested changes.

**USER'S INSTRUCTION**: "{user_instruction}"

**CORE PRINCIPLE: MAINTAIN A SINGLE, CONNECTED SHAPE**
- The most critical rule is that the final silhouette must be **one continuous, connected black shape**.
- All modifications must be integrated so the shape remains a single unified whole, unless the user's text instruction explicitly states otherwise.

**MANDATORY OUTPUT REQUIREMENTS**

1.  **Color & Form**:
    - **Exclusively Pure Black (#000000) on Pure White (#FFFFFF)**.
    - **Absolutely no red, gray, gradients, or anti-aliasing.** The output must be crisp and binary.

2.  **Fidelity to Instructions**:
    - Faithfully implement **all changes** shown in the red markup and described in the text.
    - If a visual mark and the text instruction conflict, **prioritize the text instruction**.
    - Preserve all areas of the original silhouette not indicated for change.

3.  **Aesthetic & Quality**:
    - Maintain the original's style, geometric simplicity, and 2D icon-like aesthetic.
    - Ensure sharp, clean edges and a well-balanced, centered composition.
    - Deliver a **final, professional-grade image** with no visible markup, helper lines, or artifacts, ready for use as a logo or icon.

**INTERPRET THE INTENT, NOT THE PRECISION**: 
Red marks are hand-drawn and may be imperfect (wobbly lines, irregular circles, draft fillings). Understand what the user MEANS, not the exact pixels.

**FINAL OUTPUT**
Generate the complete, modified, and connected black and white silhouette image now."""