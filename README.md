# Proof-of-Concept: Wine Review Summarization Service
### Objective
To explore the feasibility of using generative AI to create summarized wine reviews for an online wine retailer, making it easier for customers to understand wine quality and characteristics without reading dozens of individual reviews.

### Methodology
1. Data Preparation
    - Loaded and cleaned a raw wine review dataset (50,000 sampled records from ~17.8M lines).
    - Removed duplicates, missing values, and unnecessary columns.
    - Converted review scores into rating bands (e.g., `"85-89 (Very Good)"`).

2. Grouping Reviews
    - Reviews were grouped by wine variant and rating band.
    - Only groups with at least 5 reviews were considered for summarization to ensure meaningful content.

3. Summarization Pipeline
    - Used the Facebook BART Large CNN model to generate summaries.
    - Each group of reviews was chunked to handle token limits (max 900 tokens per chunk).
    - Summaries were generated with `min_length=30` and `max_length=80` to balance detail and conciseness.
    - Multiple chunks per group were re-summarized to produce a final coherent summary.

### Results
**Sample Summaries (5 Groups):**

1. “Yquem is one of the most sought-after wines in the world. This year's vintage was the first to be released in the UK since the 1980s.” - _[Sémillon-Sauvignon Blanc Blend, 95-100 (Perfect)]_

2. “Wonderful color, slightly dry and crisp, but couldn’t get past the "cat pee" on the finish. After a few sips, it just wouldn’t fade and even with food it was lingering. The major flavor remaining was sadly oak.” - _[Sémillon-Sauvignon Blanc Blend, 70-79 (Average)]_

3. “This is what all "premium" wines should deliver - a delicious drinking experience that warrants the $$$ required for purchase. Opened and let breathe in bottle for around 3 hours, then consumed over the next 2. This was probably the best CA cab (and one of the best wines) I’ve ever tasted. The 2003 Abreu Madrona Ranch is an excellent wine. Foley’s ’05 match baconbrook is 13.2% Cabernet Sauvignon. Riedel Sommelier Series Grand Cru Bordeaux is available in bottle for $330.” - _[Cabernet Sauvignon, 95-100 (Perfect)]_

4. “Citrus and mineral white with an almost pithy, bitter edge to the finish. Perfect companion to peel and eat shrimp or cracked crab. Lemon scented, fresh with good body and nice phenolics.” - _[Picpoul Blanc, 85-89 (Very Good)]_

5. “This is a far more tannic and acidic wine than I expected. Right now it tastes like a Pinot barrel sample. An amazing 15.8% alcohol here as a result of the 2003 heat wave. An incredible nosegay of geraniums and mint. Lots of strawberries and red fruits. Easy drinking and enjoyable.” - _[Gamay, 85-89 (Very Good)]_


**Performance:**
- Total runtime: ~6 minutes for 50,000 records.
- Summary generation took ~5.5 minutes.
- Observed a warning about token limits for very long review groups (e.g., sequences >1024 tokens), mitigated by chunking.


### Observations & Commentary

- Summaries were coherent and informative, often capturing key tasting notes and wine characteristics.
- Chunking helped handle longer groups but occasionally caused minor repetition.
- Sampling 50,000 records and filtering groups reduced computational cost while maintaining meaningful coverage.
- The approach demonstrates that generative AI can create customer-facing summaries for wine reviews efficiently.


### Next Steps / Recommendations

1. Scalability
    - Consider summarizing all groups using batching and GPU acceleration for production.
    - Explore longer-context models to handle reviews with >1024 tokens without chunking.

2. Quality & Usability
    - Test summaries with actual users for readability and usefulness.
    - Include metadata like wine year, region, or alcohol content in summaries for richer guidance.

3. Automation & Updates
    - Integrate a pipeline to automatically update summaries as new reviews are added.

4. Additional Enhancements
    - Tailor summaries for different customer profiles (e.g., novice vs. expert).
    - Highlight positive/negative sentiment in a structured way (e.g., taste, aroma, finish).

