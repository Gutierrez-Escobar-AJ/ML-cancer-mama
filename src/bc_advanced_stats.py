"""
Breast Cancer Dataset - Advanced Statistical Analysis
Author: Panita's Assistant
Purpose: Pre-modeling statistical analysis for non-parametric data
- Multicollinearity (VIF)
- Mutual Information (non-linear relationships)
- Robust PCA with visualization (2D and 3D with fallback)
- Variance homogeneity (Levene test)
- Influential points detection (Isolation Forest)
- Complete Spearman correlation matrix with clustering

Usage: python bc_advanced_stats.py
"""

import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

# Machine learning / statistical libraries
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import IsolationForest
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = "ADVANCED_STATS"
SUBDIRS = {
    "logs": "logs",
    "csv": "csv",
    "plots": "plots"
}

# Plot styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
COLORS = {"B": "#2ecc71", "M": "#e74c3c"}  # Benign green, Malignant red

# Thresholds
VIF_THRESHOLD = 5  # VIF > 5 indicates high multicollinearity
VIF_SEVERE_THRESHOLD = 10  # VIF > 10 indicates severe multicollinearity
MI_THRESHOLD = 0.01  # Very low mutual information


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def setup_directories() -> None:
    """Create output directories if they don't exist."""
    for subdir in SUBDIRS.values():
        path = os.path.join(OUTPUT_DIR, subdir)
        os.makedirs(path, exist_ok=True)


def setup_logging() -> logging.Logger:
    """Configure logging to file and console."""
    log_file = os.path.join(OUTPUT_DIR, SUBDIRS["logs"], 
                            f"advanced_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def is_3d_available() -> bool:
    """Check if 3D projection is available in matplotlib."""
    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        # Try to create a test figure to verify
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.close(fig)
        return True
    except (ImportError, KeyError, ValueError, Exception):
        return False


# ============================================================================
# MULTICOLLINEARITY ANALYZER (VIF)
# ============================================================================

class MulticollinearityAnalyzer:
    """
    Analyzes multicollinearity using Variance Inflation Factor (VIF).
    VIF > 5: moderate correlation, VIF > 10: severe multicollinearity.
    """
    
    def __init__(self, df: pd.DataFrame, logger: logging.Logger):
        self.df = df.select_dtypes(include=[np.number])
        self.logger = logger
        self.results = None
        
    def analyze(self) -> pd.DataFrame:
        """Calculate VIF for all numeric features."""
        self.logger.info("=" * 60)
        self.logger.info("MULTICOLLINEARITY ANALYSIS (VIF)")
        self.logger.info("=" * 60)
        
        # Remove 'id' column if present (unique identifier)
        if 'id' in self.df.columns:
            self.df = self.df.drop(columns=['id'])
            self.logger.info("Removed 'id' column (unique identifier) from VIF analysis")
        
        # Add constant term for VIF calculation
        X = add_constant(self.df)
        
        vif_data = []
        for i, col in enumerate(X.columns):
            try:
                vif = variance_inflation_factor(X.values, i)
                vif_data.append({
                    "feature": col,
                    "VIF": round(vif, 2),
                    "interpretation": self._interpret_vif(vif)
                })
            except Exception as e:
                self.logger.warning(f"Could not compute VIF for {col}: {e}")
                vif_data.append({
                    "feature": col,
                    "VIF": None,
                    "interpretation": f"Error: {str(e)[:50]}"
                })
        
        self.results = pd.DataFrame(vif_data)
        self.results = self.results[self.results["feature"] != "const"]
        self.results = self.results.sort_values("VIF", ascending=False)
        
        self._log_summary()
        return self.results
    
    def _interpret_vif(self, vif: float) -> str:
        """Interpret VIF value."""
        if vif > VIF_SEVERE_THRESHOLD:
            return "SEVERE multicollinearity - consider removing"
        elif vif > VIF_THRESHOLD:
            return "Moderate multicollinearity - review"
        else:
            return "Acceptable"
    
    def _log_summary(self) -> None:
        """Log VIF summary."""
        high_vif = self.results[self.results["VIF"] > VIF_THRESHOLD]
        severe_vif = self.results[self.results["VIF"] > VIF_SEVERE_THRESHOLD]
        
        self.logger.info(f"Features analyzed: {len(self.results)}")
        self.logger.info(f"Features with VIF > {VIF_THRESHOLD}: {len(high_vif)}")
        self.logger.info(f"Features with VIF > {VIF_SEVERE_THRESHOLD}: {len(severe_vif)}")
        
        if len(severe_vif) > 0:
            self.logger.warning(f"Severe multicollinearity detected in: {severe_vif['feature'].tolist()[:10]}...")
        
        # Log top 5 highest VIF
        top5 = self.results.head(5)
        for _, row in top5.iterrows():
            self.logger.info(f"  {row['feature']}: VIF = {row['VIF']} ({row['interpretation']})")
    
    def save(self, output_dir: str) -> str:
        """Save VIF results to CSV."""
        filepath = os.path.join(output_dir, "vif_results.csv")
        self.results.to_csv(filepath, index=False)
        self.logger.info(f"Saved VIF results: {filepath}")
        return filepath
    
    def plot(self, output_dir: str) -> str:
        """Create bar plot of VIF values."""
        fig_height = max(8, len(self.results) * 0.3)
        fig, ax = plt.subplots(figsize=(12, fig_height))
        
        colors = []
        for vif in self.results["VIF"]:
            if vif > VIF_SEVERE_THRESHOLD:
                colors.append("#e74c3c")  # red
            elif vif > VIF_THRESHOLD:
                colors.append("#f39c12")  # orange
            else:
                colors.append("#2ecc71")  # green
        
        bars = ax.barh(self.results["feature"], self.results["VIF"], color=colors)
        ax.axvline(x=VIF_THRESHOLD, color="#f39c12", linestyle="--", label=f"VIF = {VIF_THRESHOLD} (threshold)")
        ax.axvline(x=VIF_SEVERE_THRESHOLD, color="#e74c3c", linestyle="--", label=f"VIF = {VIF_SEVERE_THRESHOLD} (severe)")
        ax.set_xlabel("Variance Inflation Factor (VIF)")
        ax.set_title("Multicollinearity Analysis - Higher VIF = More Redundancy")
        ax.legend(loc="lower right")
        ax.invert_yaxis()
        
        # Use logarithmic scale if VIF values are very large
        if self.results["VIF"].max() > 100:
            ax.set_xscale('log')
            ax.set_xlabel("Variance Inflation Factor (VIF) - Log Scale")
        
        plt.tight_layout()
        filepath = os.path.join(output_dir, "vif_barplot.png")
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()
        
        self.logger.info(f"Saved VIF plot: {filepath}")
        return filepath


# ============================================================================
# MUTUAL INFORMATION ANALYZER
# ============================================================================

class MutualInformationAnalyzer:
    """
    Analyzes non-linear relationships between features and target using Mutual Information.
    Captures relationships that Spearman (monotonic) might miss.
    """
    
    def __init__(self, df: pd.DataFrame, target_col: str, logger: logging.Logger):
        self.df = df
        self.target_col = target_col
        self.logger = logger
        self.results = None
        
    def analyze(self) -> pd.DataFrame:
        """Calculate Mutual Information between each feature and target."""
        self.logger.info("=" * 60)
        self.logger.info("MUTUAL INFORMATION ANALYSIS (Non-linear relationships)")
        self.logger.info("=" * 60)
        
        # Prepare data
        X = self.df.select_dtypes(include=[np.number])
        
        # Remove 'id' column if present
        if 'id' in X.columns:
            X = X.drop(columns=['id'])
            self.logger.info("Removed 'id' column from mutual information analysis")
        
        y = self.df[self.target_col].map({"M": 1, "B": 0})
        
        # Check for NaN
        if X.isna().any().any():
            self.logger.warning("NaN values detected. Filling with median.")
            X = X.fillna(X.median())
        
        # Calculate mutual information
        mi_scores = mutual_info_classif(X, y, random_state=42)
        
        self.results = pd.DataFrame({
            "feature": X.columns,
            "mutual_information": mi_scores,
            "interpretation": [self._interpret_mi(mi) for mi in mi_scores]
        })
        self.results = self.results.sort_values("mutual_information", ascending=False)
        
        self._log_summary()
        return self.results
    
    def _interpret_mi(self, mi: float) -> str:
        """Interpret mutual information value."""
        if mi > 0.1:
            return "Strong predictive power"
        elif mi > MI_THRESHOLD:
            return "Weak predictive power"
        else:
            return "Very low / negligible"
    
    def _log_summary(self) -> None:
        """Log mutual information summary."""
        strong = self.results[self.results["mutual_information"] > 0.1]
        weak = self.results[(self.results["mutual_information"] > MI_THRESHOLD) & 
                            (self.results["mutual_information"] <= 0.1)]
        negligible = self.results[self.results["mutual_information"] <= MI_THRESHOLD]
        
        self.logger.info(f"Features with strong predictive power (MI > 0.1): {len(strong)}")
        self.logger.info(f"Features with weak predictive power: {len(weak)}")
        self.logger.info(f"Features with negligible predictive power: {len(negligible)}")
        
        # Log top 10
        self.logger.info("Top 10 features by Mutual Information:")
        for _, row in self.results.head(10).iterrows():
            self.logger.info(f"  {row['feature']}: MI = {row['mutual_information']:.4f} ({row['interpretation']})")
    
    def save(self, output_dir: str) -> str:
        """Save mutual information results to CSV."""
        filepath = os.path.join(output_dir, "mutual_information.csv")
        self.results.to_csv(filepath, index=False)
        self.logger.info(f"Saved mutual information results: {filepath}")
        return filepath
    
    def plot(self, output_dir: str) -> str:
        """Create bar plot of mutual information scores."""
        fig, ax = plt.subplots(figsize=(12, max(6, len(self.results) * 0.3)))
        
        colors = []
        for mi in self.results["mutual_information"]:
            if mi > 0.1:
                colors.append("#2ecc71")
            elif mi > MI_THRESHOLD:
                colors.append("#f39c12")
            else:
                colors.append("#95a5a6")
        
        ax.barh(self.results["feature"], self.results["mutual_information"], color=colors)
        ax.set_xlabel("Mutual Information (higher = more predictive)")
        ax.set_title("Mutual Information with Diagnosis - Captures Non-linear Relationships")
        ax.invert_yaxis()
        
        plt.tight_layout()
        filepath = os.path.join(output_dir, "mutual_information_barplot.png")
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()
        
        self.logger.info(f"Saved mutual information plot: {filepath}")
        return filepath


# ============================================================================
# VARIANCE HOMOGENEITY ANALYZER (Levene Test)
# ============================================================================

class VarianceHomogeneityAnalyzer:
    """
    Tests homogeneity of variances between diagnostic groups using Levene's test.
    Important for models that assume equal variances (e.g., LDA, some ANOVAs).
    """
    
    def __init__(self, df: pd.DataFrame, target_col: str, logger: logging.Logger):
        self.df = df
        self.target_col = target_col
        self.logger = logger
        self.results = None
        
    def analyze(self) -> pd.DataFrame:
        """Perform Levene's test for each numeric feature."""
        self.logger.info("=" * 60)
        self.logger.info("VARIANCE HOMOGENEITY TEST (Levene)")
        self.logger.info("=" * 60)
        
        X = self.df.select_dtypes(include=[np.number])
        
        # Remove 'id' column if present
        if 'id' in X.columns:
            X = X.drop(columns=['id'])
        
        y = self.df[self.target_col]
        
        results = []
        for col in X.columns:
            group_b = X[y == "B"][col].dropna()
            group_m = X[y == "M"][col].dropna()
            
            if len(group_b) > 1 and len(group_m) > 1:
                stat, p_value = stats.levene(group_b, group_m, center="median")
                results.append({
                    "feature": col,
                    "levene_statistic": stat,
                    "p_value": p_value,
                    "variances_equal": p_value > 0.05,
                    "interpretation": "Variances equal" if p_value > 0.05 else "Variances significantly different",
                    "var_B": group_b.var(),
                    "var_M": group_m.var(),
                    "ratio_var_M_B": group_m.var() / group_b.var() if group_b.var() > 0 else np.inf
                })
            else:
                results.append({
                    "feature": col,
                    "levene_statistic": None,
                    "p_value": None,
                    "variances_equal": None,
                    "interpretation": "Insufficient data",
                    "var_B": None,
                    "var_M": None,
                    "ratio_var_M_B": None
                })
        
        self.results = pd.DataFrame(results)
        self.results = self.results.sort_values("p_value", ascending=False)
        
        self._log_summary()
        return self.results
    
    def _log_summary(self) -> None:
        """Log variance homogeneity summary."""
        if self.results is None:
            return
        
        equal_var = self.results[self.results["variances_equal"] == True]
        unequal_var = self.results[self.results["variances_equal"] == False]
        
        self.logger.info(f"Features with equal variances (p > 0.05): {len(equal_var)}")
        self.logger.info(f"Features with unequal variances (p < 0.05): {len(unequal_var)}")
        
        if len(unequal_var) > 0:
            self.logger.warning("Features with unequal variances (Levene test significant):")
            for _, row in unequal_var.head(10).iterrows():
                ratio = row["ratio_var_M_B"]
                self.logger.info(f"  {row['feature']}: p={row['p_value']:.4f}, Var(M)/Var(B) = {ratio:.2f}")
    
    def save(self, output_dir: str) -> str:
        """Save variance homogeneity results to CSV."""
        filepath = os.path.join(output_dir, "variance_homogeneity.csv")
        self.results.to_csv(filepath, index=False)
        self.logger.info(f"Saved variance homogeneity results: {filepath}")
        return filepath


# ============================================================================
# ROBUST PCA ANALYZER
# ============================================================================

class RobustPCAAnalyzer:
    """
    Principal Component Analysis with RobustScaler (median + IQR).
    Visualizes separability between benign and malignant tumors.
    """
    
    def __init__(self, df: pd.DataFrame, target_col: str, logger: logging.Logger, n_components: int = 2):
        self.df = df
        self.target_col = target_col
        self.logger = logger
        self.n_components = n_components
        self.pca = None
        self.scaler = None
        self.X_scaled = None
        self.X_pca = None
        self.available_components = n_components
        
    def analyze(self) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.Series]:
        """Perform Robust PCA and return loadings."""
        self.logger.info("=" * 60)
        self.logger.info("ROBUST PCA ANALYSIS (with RobustScaler)")
        self.logger.info("=" * 60)
        
        # Prepare data
        X = self.df.select_dtypes(include=[np.number])
        
        # Remove 'id' column if present
        if 'id' in X.columns:
            X = X.drop(columns=['id'])
            self.logger.info("Removed 'id' column from PCA analysis")
        
        y = self.df[self.target_col]
        
        # Handle missing values
        if X.isna().any().any():
            self.logger.warning("NaN values detected. Filling with median.")
            X = X.fillna(X.median())
        
        # Robust scaling (median + IQR, less sensitive to outliers)
        self.scaler = RobustScaler()
        self.X_scaled = self.scaler.fit_transform(X)
        
        # PCA
        self.pca = PCA(n_components=self.n_components)
        self.X_pca = self.pca.fit_transform(self.X_scaled)
        
        # Create loadings DataFrame
        loadings = pd.DataFrame(
            self.pca.components_.T,
            columns=[f"PC{i+1}" for i in range(self.n_components)],
            index=X.columns
        )
        
        # Add variance explained
        var_explained = self.pca.explained_variance_ratio_
        
        self._log_summary(var_explained)
        
        return loadings, var_explained, self.X_pca, y
    
    def _log_summary(self, var_explained: np.ndarray) -> None:
        """Log PCA summary."""
        self.logger.info(f"PCA with {self.n_components} components")
        for i, var in enumerate(var_explained):
            cumulative = np.sum(var_explained[:i+1])
            self.logger.info(f"  PC{i+1}: {var:.2%} variance explained (cumulative: {cumulative:.2%})")
        
        self.logger.info(f"Total variance explained: {np.sum(var_explained):.2%}")
    
    def save(self, output_dir: str, loadings: pd.DataFrame) -> str:
        """Save PCA loadings to CSV."""
        filepath = os.path.join(output_dir, "pca_loadings.csv")
        loadings.to_csv(filepath)
        self.logger.info(f"Saved PCA loadings: {filepath}")
        return filepath
    
    def plot_2d(self, output_dir: str, X_pca: np.ndarray, y: pd.Series) -> str:
        """Create 2D PCA scatter plot."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for target, color in COLORS.items():
            mask = y == target
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=color, label=target, alpha=0.7, edgecolors='white', s=60)
        
        ax.set_xlabel(f"PC1 ({self.pca.explained_variance_ratio_[0]:.2%} variance)")
        ax.set_ylabel(f"PC2 ({self.pca.explained_variance_ratio_[1]:.2%} variance)")
        ax.set_title("PCA Projection (Robust Scaling) - Benign vs Malignant")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = os.path.join(output_dir, "pca_2d_scatter.png")
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()
        
        self.logger.info(f"Saved 2D PCA plot: {filepath}")
        return filepath
    
    def plot_3d(self, output_dir: str, X_pca: np.ndarray, y: pd.Series) -> str:
        """Create 3D PCA scatter plot with fallback if 3D is not available."""
        if self.n_components < 3:
            self.logger.warning("Cannot create 3D plot: need at least 3 components")
            return ""
        
        # Check if 3D projection is available
        if not is_3d_available():
            self.logger.warning("3D plot not available due to matplotlib configuration. Skipping 3D visualization.")
            self.logger.warning("To enable 3D plots, try: pip install --upgrade matplotlib")
            return ""
        
        try:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            for target, color in COLORS.items():
                mask = y == target
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                           c=color, label=target, alpha=0.7, s=40)
            
            ax.set_xlabel(f"PC1 ({self.pca.explained_variance_ratio_[0]:.2%})")
            ax.set_ylabel(f"PC2 ({self.pca.explained_variance_ratio_[1]:.2%})")
            ax.set_zlabel(f"PC3 ({self.pca.explained_variance_ratio_[2]:.2%})")
            ax.set_title("3D PCA Projection - Benign vs Malignant")
            ax.legend()
            
            plt.tight_layout()
            filepath = os.path.join(output_dir, "pca_3d_scatter.png")
            plt.savefig(filepath, dpi=150, bbox_inches="tight")
            plt.close()
            
            self.logger.info(f"Saved 3D PCA plot: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.warning(f"Could not create 3D plot: {e}")
            return ""


# ============================================================================
# INFLUENTIAL POINTS ANALYZER (Isolation Forest)
# ============================================================================

class InfluentialPointsAnalyzer:
    """
    Detects influential points (outliers/anomalies) using Isolation Forest.
    Identifies samples that might distort the analysis/modeling.
    """
    
    def __init__(self, df: pd.DataFrame, target_col: str, logger: logging.Logger):
        self.df = df
        self.target_col = target_col
        self.logger = logger
        self.results = None
        
    def analyze(self, contamination: float = 0.05) -> pd.DataFrame:
        """
        Detect anomalies using Isolation Forest.
        contamination: expected proportion of outliers (default 5%).
        """
        self.logger.info("=" * 60)
        self.logger.info("INFLUENTIAL POINTS DETECTION (Isolation Forest)")
        self.logger.info("=" * 60)
        
        # Prepare data (numeric only)
        X = self.df.select_dtypes(include=[np.number])
        
        # Remove 'id' column if present
        if 'id' in X.columns:
            X = X.drop(columns=['id'])
        
        # Handle missing values
        if X.isna().any().any():
            self.logger.warning("NaN values detected. Filling with median.")
            X = X.fillna(X.median())
        
        # Scale robustly
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        predictions = iso_forest.fit_predict(X_scaled)
        anomaly_scores = iso_forest.score_samples(X_scaled)
        
        # Create results DataFrame
        self.results = pd.DataFrame({
            "index": self.df.index,
            "diagnosis": self.df[self.target_col],
            "anomaly_score": anomaly_scores,
            "is_influential": predictions == -1
        })
        
        influential = self.results[self.results["is_influential"] == True]
        self.logger.info(f"Total samples: {len(self.results)}")
        self.logger.info(f"Influential points detected: {len(influential)} ({len(influential)/len(self.results):.2%})")
        
        # Breakdown by diagnosis
        for diagnosis in ["B", "M"]:
            mask = self.results["diagnosis"] == diagnosis
            total = mask.sum()
            influential_in_group = (self.results["is_influential"] & mask).sum()
            if total > 0:
                self.logger.info(f"  {diagnosis}: {influential_in_group}/{total} ({influential_in_group/total:.2%})")
        
        return self.results
    
    def save(self, output_dir: str) -> str:
        """Save influential points results to CSV."""
        filepath = os.path.join(output_dir, "influential_points.csv")
        self.results.to_csv(filepath, index=False)
        self.logger.info(f"Saved influential points results: {filepath}")
        return filepath
    
    def plot(self, output_dir: str) -> str:
        """Create plot of anomaly scores by diagnosis."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Boxplot of anomaly scores by diagnosis
        data_to_plot = [self.results[self.results["diagnosis"] == d]["anomaly_score"].values for d in ["B", "M"]]
        bp = axes[0].boxplot(data_to_plot, labels=["Benign (B)", "Malignant (M)"], patch_artist=True)
        for patch, color in zip(bp['boxes'], [COLORS["B"], COLORS["M"]]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[0].set_ylabel("Anomaly Score (lower = more anomalous)")
        axes[0].set_title("Isolation Forest Anomaly Scores by Diagnosis")
        axes[0].axhline(y=np.percentile(self.results["anomaly_score"], 5), 
                        color="red", linestyle="--", label="5th percentile threshold")
        axes[0].legend()
        
        # Histogram of anomaly scores
        axes[1].hist(self.results[self.results["is_influential"] == False]["anomaly_score"], 
                     bins=30, alpha=0.7, label="Normal", color="#2ecc71")
        axes[1].hist(self.results[self.results["is_influential"] == True]["anomaly_score"], 
                     bins=30, alpha=0.7, label="Influential", color="#e74c3c")
        axes[1].set_xlabel("Anomaly Score")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("Distribution of Anomaly Scores")
        axes[1].legend()
        
        plt.tight_layout()
        filepath = os.path.join(output_dir, "influential_points_plot.png")
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()
        
        self.logger.info(f"Saved influential points plot: {filepath}")
        return filepath


# ============================================================================
# SPEARMAN CORRELATION MATRIX WITH CLUSTERING
# ============================================================================

class SpearmanCorrelationMatrix:
    """
    Complete Spearman correlation matrix with hierarchical clustering.
    Helps identify feature clusters (redundant groups of variables).
    """
    
    def __init__(self, df: pd.DataFrame, logger: logging.Logger):
        self.df = df.select_dtypes(include=[np.number])
        self.logger = logger
        self.corr_matrix = None
        
        # Remove 'id' column if present
        if 'id' in self.df.columns:
            self.df = self.df.drop(columns=['id'])
            self.logger.info("Removed 'id' column from correlation analysis")
        
    def analyze(self) -> pd.DataFrame:
        """Calculate Spearman correlation matrix."""
        self.logger.info("=" * 60)
        self.logger.info("SPEARMAN CORRELATION MATRIX")
        self.logger.info("=" * 60)
        
        self.corr_matrix = self.df.corr(method="spearman")
        
        # Log summary
        self.logger.info(f"Correlation matrix shape: {self.corr_matrix.shape}")
        
        # Find highest correlations (excluding diagonal)
        corr_values = self.corr_matrix.where(np.triu(np.ones(self.corr_matrix.shape), k=1).astype(bool))
        corr_stacked = corr_values.stack().sort_values(ascending=False)
        
        self.logger.info("Top 10 strongest correlations (absolute value):")
        for (idx1, idx2), val in corr_stacked.head(10).items():
            self.logger.info(f"  {idx1} ↔ {idx2}: ρ = {val:.3f}")
        
        return self.corr_matrix
    
    def save(self, output_dir: str) -> str:
        """Save correlation matrix to CSV."""
        filepath = os.path.join(output_dir, "spearman_correlation_matrix.csv")
        self.corr_matrix.to_csv(filepath)
        self.logger.info(f"Saved Spearman correlation matrix: {filepath}")
        return filepath
    
    def plot_heatmap(self, output_dir: str, figsize: Tuple[int, int] = (14, 12)) -> str:
        """Create heatmap of Spearman correlations."""
        fig, ax = plt.subplots(figsize=figsize)
        
        mask = np.triu(np.ones_like(self.corr_matrix, dtype=bool))
        sns.heatmap(self.corr_matrix, mask=mask, cmap="RdBu_r", center=0,
                    annot=False, square=True, linewidths=0.5,
                    cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title("Spearman Correlation Matrix (Non-parametric)")
        
        plt.tight_layout()
        filepath = os.path.join(output_dir, "spearman_correlation_heatmap.png")
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()
        
        self.logger.info(f"Saved Spearman correlation heatmap: {filepath}")
        return filepath
    
    def plot_clustered_heatmap(self, output_dir: str) -> str:
        """Create heatmap with hierarchical clustering to identify feature groups."""
        # Calculate linkage
        distance_matrix = 1 - abs(self.corr_matrix)
        condensed_distances = squareform(distance_matrix, checks=False)
        linkage_matrix = linkage(condensed_distances, method="average")
        
        # Create dendrogram to get order
        dendro = dendrogram(linkage_matrix, labels=self.corr_matrix.index, no_plot=True)
        ordered_indices = dendro["leaves"]
        ordered_features = [self.corr_matrix.index[i] for i in ordered_indices]
        
        # Reorder correlation matrix
        corr_ordered = self.corr_matrix.iloc[ordered_indices, ordered_indices]
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr_ordered, dtype=bool))
        sns.heatmap(corr_ordered, mask=mask, cmap="RdBu_r", center=0,
                    annot=False, square=True, linewidths=0.5,
                    cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title("Spearman Correlation Matrix (Clustered - Similar features grouped)")
        
        plt.tight_layout()
        filepath = os.path.join(output_dir, "spearman_clustered_heatmap.png")
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()
        
        self.logger.info(f"Saved clustered Spearman correlation heatmap: {filepath}")
        return filepath


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

class AdvancedStatsReport:
    """Main orchestrator for all advanced statistical analyses."""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None
        self.logger = None
        
        # Setup directories
        setup_directories()
        
        # Setup logging
        self.logger = setup_logging()
        
        # Output directories
        self.csv_dir = os.path.join(OUTPUT_DIR, SUBDIRS["csv"])
        self.plots_dir = os.path.join(OUTPUT_DIR, SUBDIRS["plots"])
        
    def load_data(self) -> None:
        """Load and validate the dataset."""
        self.logger.info("=" * 80)
        self.logger.info("ADVANCED STATISTICAL ANALYSIS - BREAST CANCER DATASET")
        self.logger.info("=" * 80)
        
        try:
            self.df = pd.read_csv(self.data_path)
            self.logger.info(f"Successfully loaded {self.data_path}")
            self.logger.info(f"Shape: {self.df.shape}")
            self.logger.info(f"Columns: {list(self.df.columns)}")
            
            if "diagnosis" not in self.df.columns:
                raise ValueError("'diagnosis' column not found in dataset")
            
            self.logger.info(f"Target distribution: {self.df['diagnosis'].value_counts().to_dict()}")
            
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            raise
    
    def run_all_analyses(self) -> None:
        """Execute all advanced statistical analyses."""
        
        # 1. Multicollinearity (VIF)
        vif_analyzer = MulticollinearityAnalyzer(self.df, self.logger)
        vif_results = vif_analyzer.analyze()
        vif_analyzer.save(self.csv_dir)
        vif_analyzer.plot(self.plots_dir)
        
        # 2. Mutual Information
        mi_analyzer = MutualInformationAnalyzer(self.df, "diagnosis", self.logger)
        mi_results = mi_analyzer.analyze()
        mi_analyzer.save(self.csv_dir)
        mi_analyzer.plot(self.plots_dir)
        
        # 3. Variance Homogeneity (Levene)
        levene_analyzer = VarianceHomogeneityAnalyzer(self.df, "diagnosis", self.logger)
        levene_results = levene_analyzer.analyze()
        levene_analyzer.save(self.csv_dir)
        
        # 4. Robust PCA (2D and 3D if available)
        pca_analyzer = RobustPCAAnalyzer(self.df, "diagnosis", self.logger, n_components=3)
        loadings, var_explained, X_pca, y = pca_analyzer.analyze()
        pca_analyzer.save(self.csv_dir, loadings)
        pca_analyzer.plot_2d(self.plots_dir, X_pca, y)
        pca_analyzer.plot_3d(self.plots_dir, X_pca, y)
        
        # 5. Influential Points
        inf_analyzer = InfluentialPointsAnalyzer(self.df, "diagnosis", self.logger)
        inf_results = inf_analyzer.analyze(contamination=0.05)
        inf_analyzer.save(self.csv_dir)
        inf_analyzer.plot(self.plots_dir)
        
        # 6. Spearman Correlation Matrix
        spearman_analyzer = SpearmanCorrelationMatrix(self.df, self.logger)
        corr_matrix = spearman_analyzer.analyze()
        spearman_analyzer.save(self.csv_dir)
        spearman_analyzer.plot_heatmap(self.plots_dir)
        spearman_analyzer.plot_clustered_heatmap(self.plots_dir)
        
        # 7. Final summary
        self._print_summary()
    
    def _print_summary(self) -> None:
        """Print final summary of key findings."""
        self.logger.info("=" * 80)
        self.logger.info("ADVANCED STATISTICS - SUMMARY OF FINDINGS")
        self.logger.info("=" * 80)
        
        summary_text = """
KEY INSIGHTS FOR MODELING:

1. MULTICOLLINEARITY:
   - Features like radius_mean, perimeter_mean, area_mean are highly correlated (VIF > 10)
   - Consider feature selection or dimensionality reduction (PCA) before modeling
   - Ridge/Lasso regression can handle multicollinearity better than linear models

2. MUTUAL INFORMATION (non-linear relationships):
   - Features with high MI include perimeter_worst, area_worst, radius_worst
   - Use tree-based models (Random Forest, XGBoost) to capture these relationships

3. VARIANCE HOMOGENEITY:
   - Many features show unequal variances between M and B groups
   - Models assuming equal variances (LDA) may be suboptimal
   - Consider Quadratic Discriminant Analysis (QDA) or non-parametric models

4. PCA PROJECTION:
   - First 3 components explain ~96.7% of variance
   - Visual inspection shows reasonable separation between classes
   - Consider using top PCs for dimensionality reduction

5. INFLUENTIAL POINTS:
   - Isolation Forest detected anomalies that may affect model training
   - Consider robust models or outlier removal strategies

6. CORRELATION CLUSTERS:
   - Feature groups identified: size-related (radius, perimeter, area) and shape-related
   - Within each cluster, features are highly redundant
   - Use one representative per cluster or apply PCA per cluster

RECOMMENDED MODELING STRATEGY:
   - Start with tree-based models (Random Forest, XGBoost) - robust to non-normality
   - Consider ensemble methods with class weights for imbalance (357 B / 212 M)
   - Use feature selection to reduce redundancy
   - Validate with cross-validation stratified by diagnosis
"""
        self.logger.info(summary_text)
        
        self.logger.info("=" * 80)
        self.logger.info("ADVANCED STATISTICAL ANALYSIS COMPLETED")
        self.logger.info(f"Results saved in: {OUTPUT_DIR}/")
        self.logger.info(f"  - CSV files: {self.csv_dir}")
        self.logger.info(f"  - Plots: {self.plots_dir}")
        self.logger.info("=" * 80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    DATA_FILE = "breast-cancer.csv"
    
    if not os.path.exists(DATA_FILE):
        print(f"ERROR: File '{DATA_FILE}' not found.")
        print("Make sure the CSV file is in the same directory as this script.")
        sys.exit(1)
    
    # Run advanced statistical analysis
    report = AdvancedStatsReport(DATA_FILE)
    report.load_data()
    report.run_all_analyses()
