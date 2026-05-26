"""
Prompt templates for Flipbook-style generation.
Forces adaptive, illustrative style + viewport awareness.
"""

# Base template for image generation
IMAGE_PROMPT_TEMPLATE = """You are generating a full-screen illustrative image for an infinite visual browser called Flipbook.

CONTEXT: {context}

USER ACTION: {user_action}

TASK: Create a detailed, visually rich image that:
1. Explores the topic in depth with accurate, grounded information
2. Uses an illustrative/editorial style appropriate for educational content
3. Renders ALL text within the image as painted pixels (no text overlays)
4. Fills the entire viewport (16:9 or 4:3 depending on screen)
5. Uses visual hierarchy — main subject large and central, details around edges
6. Includes subtle depth cues and compositional balance

STYLE GUIDANCE:
- Rich, detailed, cinematic lighting
- Editorial illustration meets infographic
- No UI chrome, no buttons, no text labels — only image content
- Text rendered as part of the scene where appropriate (signs, books, screens, etc.)

WINDOW SIZE: {width}x{height} pixels
ASPECT RATIO: {aspect_ratio}

INFORMATION GROUNDING (from web search):
{grounding}

Generate a prompt suitable for a high-quality text-to-image model."""


# Click interpretation template
CLICK_INTERPRET_TEMPLATE = """You are analyzing a click in an infinite visual browser called Flipbook.

PREVIOUS IMAGE DESCRIPTION:
{image_description}

CLICK COORDINATES: x={x}, y={y} (normalized 0-1)

TASK: Interpret what the user is trying to explore based on:
1. WHERE they clicked (x,y position on image)
2. WHAT was likely at that location based on the image description

Respond with:
1. A short label (2-5 words) for what was clicked
2. A detailed exploration prompt (1-2 sentences) for what to generate next

Be specific — avoid generic "tell me more about X". Focus on the exact visual element clicked.

FORMAT:
label: <short label>
explore: <detailed next-step prompt>"""


# Video transition prompt template
VIDEO_TRANSITION_TEMPLATE = """You are generating a video transition for Flipbook.

FROM IMAGE: {from_description}
TO IMAGE: {to_description}

TASK: Create a prompt for LTX-Video that animates a smooth, cinematic transition.

The video should:
1. Start from the previous image's composition
2. End at the new image's starting composition
3. Use 16 frames at 24fps (~0.67 seconds)
4. Maintain visual continuity — shared elements animate smoothly
5. Be cinematic, not jerky — ease in/out

STYLE: Smooth crossfade with subtle camera movement
RESOLUTION: 768x768 (prototype) or 1024x576 (widescreen)

Generate a prompt for the video diffusion model."""


# Agent search query template
SEARCH_QUERY_TEMPLATE = """Given the user's current exploration topic: {topic}

Generate a focused web search query that will find factual, up-to-date information.

Requirements:
- 1-3 sentences
- Factual and specific, not vague
- Include relevant context from the conversation history if available

Return only the search query string."""


# Grounding prompt for agent results
GROUNDING_TEMPLATE = """You are grounding an image generation prompt with real web search results.

TOPIC: {topic}

SEARCH RESULTS:
{search_results}

TASK: Extract the 3-5 most important factual details from these results that would make the image accurate and informative.

Format as bullet points — factual, specific, numerical where possible.
Do not hallucinate. Only use information from the search results or your world knowledge if it matches."""

