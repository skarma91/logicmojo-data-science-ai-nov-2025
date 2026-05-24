# Project Report: Wholesale Customers Clustering

## 1. Problem Framing & Choice of Clustering Methods

**Problem:** The goal is to segment 440 wholesale customers based on their annual spending across six product categories (Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen). Effective segmentation enables the wholesaler to tailor marketing, pricing, and inventory strategies to distinct customer groups.

**Why clustering?** Customer segmentation is an unsupervised learning problem — we have no predefined labels for customer types. Clustering algorithms discover natural groupings based on spending patterns.

**Methods chosen:**
- **K-Means:** The most widely used partitional clustering algorithm. It works well when clusters are roughly spherical and similarly sized, and scales efficiently.
- **Hierarchical (Agglomerative) Clustering:** Provides a dendrogram that visualizes the nested cluster structure, helping validate the number of clusters chosen for K-Means.
- **DBSCAN:** A density-based method that can discover arbitrarily shaped clusters and identify outliers as noise. Included to test whether the data has non-convex structure.

## 2. Key EDA Insights

1. **Heavy right skew:** All six spending features exhibit strong positive skew (2.5–11.0), with long tails of high-spending customers. This necessitated log-transformation before clustering.
2. **Strong correlations:** Grocery and Detergents_Paper are highly correlated (r ≈ 0.92), and both correlate with Milk. This indicates a "household staples" purchasing dimension.
3. **Fresh and Frozen are independent:** These features form a separate purchasing axis, suggesting a different customer type (likely HoReCa).
4. **Outliers present:** Box plots revealed numerous upper outliers in all spending categories, particularly Fresh and Grocery. Log-transformation mitigated their influence.
5. **Channel as a natural divide:** The pairplot colored by Channel (HoReCa vs. Retail) already hinted at two distinct groups, which clustering later confirmed.

## 3. Choosing the Number of Clusters

- The **Elbow Method** showed a clear bend at k=2, with diminishing returns beyond k=3.
- The **Silhouette Score** was highest at k=2 across all methods tested (K-Means and Agglomerative).
- The **dendrogram** from hierarchical clustering showed two major branches, confirming the two-cluster structure.
- We tested both k=2 and k=3; k=2 consistently yielded higher silhouette scores and more interpretable segments.

**Conclusion:** k=2 is the optimal number of clusters for this dataset.

## 4. Cluster Profiling

| Attribute | Cluster 0 — Fresh-focused (HoReCa) | Cluster 1 — Grocery-focused (Retail) |
|---|---|---|
| **Dominant categories** | Fresh, Frozen, Delicassen | Grocery, Detergents_Paper, Milk |
| **Likely channel** | Hotel/Restaurant/Cafe (Channel 1) | Retail stores (Channel 2) |
| **Buying behavior** | Perishable and specialty items in bulk | Household staples and cleaning products |
| **Customer count** | Roughly 60% of the base | Roughly 40% of the base |

The two clusters align closely with the Channel variable, validating that the spending patterns naturally separate HoReCa and Retail customers.

## 5. Visualizations Summary

- **PCA (2 components):** Captures ~70% of total variance and shows clear separation between the two clusters along the first principal component.
- **t-SNE:** Confirms the two-cluster structure with visually distinct groupings in a non-linear embedding.
- **Centroid heatmap:** Provides an intuitive view of which spending categories define each cluster.
- **Side-by-side comparison:** K-Means and Agglomerative produce nearly identical clusters; DBSCAN identifies a similar split but with some noise points.

## 6. Recommendations for Business Use

1. **Differentiated marketing:** Design separate promotional campaigns — fresh produce and frozen deals for HoReCa customers; bundle offers on grocery, milk, and detergents for Retail customers.
2. **Inventory optimization:** Allocate warehouse capacity and delivery schedules based on cluster-specific demand patterns (perishables vs. shelf-stable goods).
3. **Pricing strategy:** Offer volume-based discounts on Fresh and Frozen for Cluster 0 (high-volume HoReCa buyers), and loyalty/subscription programs for Cluster 1 (regular retail restocking).
4. **Customer retention:** Monitor spending changes over time — a HoReCa customer shifting toward retail patterns may indicate account churn or a business pivot, prompting proactive outreach.
5. **New customer onboarding:** Classify new customers into a segment early (based on initial orders) to immediately tailor the sales approach.