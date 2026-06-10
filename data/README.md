# Data Directory

This directory contains datasets for implicit meaning classification.

## Files

- **samples.json**: Full dataset with original sentences, revised versions (with implicit info in angle brackets), article names, and surrounding context.

- **gold_standard.json**: Annotated gold standard data with labels (Implicit/New Information), MACE scores, majority vote labels, and confidence distributions, if available. 
    - *Label Distribution:* Index 0 = "Implicit", Index 1 = "New Information" 
    - *Confidence Score:* Index 0 = 1-2, Index 1 = 3, Index 2 = 4-5 

- **few_shots.json**: Few-shot examples organized by sample count and label type. Includes annotation metadata and confidence distributions.

- **splits.json**: Train/Dev/Test/OOD split assignments mapping sample IDs to their dataset partition.

- **id2label.json**: Label mapping—associates sample IDs with their Implicit/New Information classification labels.

- **id2topic.json**: Topic mapping—maps sample IDs to topic categories (e.g., "health").

- **gold_standard_topics_subcategories.json**: Gold standard with additional subcategory annotations for samples from topic study.
    - *Label Distribution:* Index 0 = "Implicit", Index 1 = "New Information" 
    - *Confidence Score:* Index 0 = 1-2, Index 1 = 3, Index 2 = 4-5 
    - *Categories Distribution:* Index 0 = "Context", Index 1 = "Logical Reasoning", Index 2 = "Background Knowledge", Index 3 = "Other" 

## Data Format

Each sample contains:
- **Original**: The original sentence
- **Revised**: The same sentence with implicit information made explicit (marked with `<...>`)
- **Label**: Classification as "Implicit" or "New Information"
- **Confidence/Annotation metadata**: Annotator agreements and confidence scores
