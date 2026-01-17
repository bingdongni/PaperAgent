"""
Advanced Statistical Analysis Tools

Provides comprehensive statistical analysis capabilities including:
- Descriptive statistics
- Hypothesis testing
- Regression analysis
- Survival analysis
- Time series analysis
- Machine learning integration
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import scipy.stats as stats
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu, chi2_contingency
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.power import TTestIndPower
import pingouin as pg
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report
import plotly.graph_objects as go
import plotly.express as px
from loguru import logger


class AdvancedStatistics:
    """Advanced statistical analysis toolkit"""

    @staticmethod
    def descriptive_stats(data: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Comprehensive descriptive statistics

        Args:
            data: Input dataframe
            columns: Specific columns to analyze (None = all numeric)

        Returns:
            Dictionary with descriptive statistics
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()

        results = {}

        for col in columns:
            col_data = data[col].dropna()

            results[col] = {
                'count': len(col_data),
                'mean': float(col_data.mean()),
                'median': float(col_data.median()),
                'std': float(col_data.std()),
                'var': float(col_data.var()),
                'min': float(col_data.min()),
                'max': float(col_data.max()),
                'q25': float(col_data.quantile(0.25)),
                'q75': float(col_data.quantile(0.75)),
                'iqr': float(col_data.quantile(0.75) - col_data.quantile(0.25)),
                'skewness': float(col_data.skew()),
                'kurtosis': float(col_data.kurtosis()),
                'cv': float(col_data.std() / col_data.mean() if col_data.mean() != 0 else 0),  # Coefficient of variation
                'sem': float(col_data.sem())  # Standard error of mean
            }

        return results

    @staticmethod
    def normality_test(data: pd.Series) -> Dict[str, Any]:
        """
        Test for normality using Shapiro-Wilk test

        Args:
            data: Data series

        Returns:
            Test results
        """
        data_clean = data.dropna()

        stat, p_value = shapiro(data_clean)

        return {
            'test': 'Shapiro-Wilk',
            'statistic': float(stat),
            'p_value': float(p_value),
            'is_normal': p_value > 0.05,
            'interpretation': 'Data appears normally distributed' if p_value > 0.05 else 'Data does not appear normally distributed'
        }

    @staticmethod
    def ttest_analysis(group1: pd.Series, group2: pd.Series, paired: bool = False) -> Dict[str, Any]:
        """
        T-test analysis with effect size

        Args:
            group1: First group data
            group2: Second group data
            paired: Whether to perform paired t-test

        Returns:
            T-test results with effect size
        """
        g1 = group1.dropna()
        g2 = group2.dropna()

        if paired:
            stat, p_value = stats.ttest_rel(g1, g2)
            test_type = 'Paired t-test'
        else:
            stat, p_value = ttest_ind(g1, g2)
            test_type = 'Independent t-test'

        # Calculate Cohen's d (effect size)
        pooled_std = np.sqrt(((len(g1) - 1) * g1.std() ** 2 + (len(g2) - 1) * g2.std() ** 2) / (len(g1) + len(g2) - 2))
        cohens_d = (g1.mean() - g2.mean()) / pooled_std if pooled_std != 0 else 0

        # Effect size interpretation
        if abs(cohens_d) < 0.2:
            effect = 'negligible'
        elif abs(cohens_d) < 0.5:
            effect = 'small'
        elif abs(cohens_d) < 0.8:
            effect = 'medium'
        else:
            effect = 'large'

        return {
            'test': test_type,
            'statistic': float(stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'cohens_d': float(cohens_d),
            'effect_size': effect,
            'group1_mean': float(g1.mean()),
            'group2_mean': float(g2.mean()),
            'mean_difference': float(g1.mean() - g2.mean())
        }

    @staticmethod
    def anova_analysis(data: pd.DataFrame, dependent_var: str, independent_var: str) -> Dict[str, Any]:
        """
        One-way ANOVA analysis

        Args:
            data: Input dataframe
            dependent_var: Dependent variable column name
            independent_var: Independent variable (grouping) column name

        Returns:
            ANOVA results
        """
        # Perform ANOVA
        formula = f'{dependent_var} ~ C({independent_var})'
        model = ols(formula, data=data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        # Post-hoc test (Tukey HSD)
        tukey = pairwise_tukeyhsd(data[dependent_var], data[independent_var])

        return {
            'test': 'One-way ANOVA',
            'f_statistic': float(anova_table['F'].iloc[0]),
            'p_value': float(anova_table['PR(>F)'].iloc[0]),
            'significant': float(anova_table['PR(>F)'].iloc[0]) < 0.05,
            'anova_table': anova_table.to_dict(),
            'post_hoc': str(tukey),
            'eta_squared': float(anova_table['sum_sq'].iloc[0] / anova_table['sum_sq'].sum())  # Effect size
        }

    @staticmethod
    def correlation_analysis(data: pd.DataFrame, method: str = 'pearson') -> Dict[str, Any]:
        """
        Correlation analysis with significance testing

        Args:
            data: Input dataframe
            method: 'pearson', 'spearman', or 'kendall'

        Returns:
            Correlation matrix and p-values
        """
        numeric_data = data.select_dtypes(include=[np.number])

        if method == 'pearson':
            corr_matrix = numeric_data.corr()
            # Calculate p-values
            pval_matrix = numeric_data.corr(method=lambda x, y: stats.pearsonr(x, y)[1])
        elif method == 'spearman':
            corr_matrix = numeric_data.corr(method='spearman')
            pval_matrix = numeric_data.corr(method=lambda x, y: stats.spearmanr(x, y)[1])
        else:
            corr_matrix = numeric_data.corr(method='kendall')
            pval_matrix = pd.DataFrame(np.ones_like(corr_matrix), columns=corr_matrix.columns, index=corr_matrix.index)

        return {
            'method': method,
            'correlation_matrix': corr_matrix.to_dict(),
            'p_value_matrix': pval_matrix.to_dict(),
            'strong_correlations': AdvancedStatistics._find_strong_correlations(corr_matrix, pval_matrix)
        }

    @staticmethod
    def _find_strong_correlations(corr_matrix: pd.DataFrame, pval_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Dict]:
        """Find strong correlations"""
        strong_corr = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr = corr_matrix.iloc[i, j]
                pval = pval_matrix.iloc[i, j]

                if abs(corr) >= threshold and pval < 0.05:
                    strong_corr.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': float(corr),
                        'p_value': float(pval)
                    })

        return strong_corr

    @staticmethod
    def regression_analysis(data: pd.DataFrame, dependent_var: str, independent_vars: List[str]) -> Dict[str, Any]:
        """
        Multiple linear regression analysis

        Args:
            data: Input dataframe
            dependent_var: Dependent variable
            independent_vars: List of independent variables

        Returns:
            Regression results
        """
        # Prepare data
        y = data[dependent_var]
        X = data[independent_vars]

        # Add constant
        X = sm.add_constant(X)

        # Fit model
        model = sm.OLS(y, X).fit()

        return {
            'r_squared': float(model.rsquared),
            'adj_r_squared': float(model.rsquared_adj),
            'f_statistic': float(model.fvalue),
            'f_pvalue': float(model.f_pvalue),
            'coefficients': model.params.to_dict(),
            'p_values': model.pvalues.to_dict(),
            'std_errors': model.bse.to_dict(),
            'confidence_intervals': model.conf_int().to_dict(),
            'summary': str(model.summary())
        }

    @staticmethod
    def power_analysis(effect_size: float, alpha: float = 0.05, power: float = 0.8) -> Dict[str, Any]:
        """
        Statistical power analysis for sample size determination

        Args:
            effect_size: Expected effect size (Cohen's d)
            alpha: Significance level
            power: Desired statistical power

        Returns:
            Required sample size
        """
        analysis = TTestIndPower()
        sample_size = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power)

        return {
            'effect_size': effect_size,
            'alpha': alpha,
            'power': power,
            'required_sample_size_per_group': int(np.ceil(sample_size)),
            'total_sample_size': int(np.ceil(sample_size * 2))
        }

    @staticmethod
    def machine_learning_analysis(
        data: pd.DataFrame,
        target: str,
        features: List[str],
        task: str = 'regression',
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """
        Machine learning analysis

        Args:
            data: Input dataframe
            target: Target variable
            features: Feature columns
            task: 'regression' or 'classification'
            test_size: Test set proportion

        Returns:
            ML model results
        """
        X = data[features]
        y = data[target]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if task == 'regression':
            # Random Forest Regressor
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)

            y_pred = model.predict(X_test_scaled)

            results = {
                'task': 'regression',
                'model': 'Random Forest Regressor',
                'r2_score': float(r2_score(y_test, y_pred)),
                'mse': float(mean_squared_error(y_test, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
                'feature_importance': dict(zip(features, model.feature_importances_.tolist()))
            }
        else:
            # Random Forest Classifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)

            y_pred = model.predict(X_test_scaled)

            results = {
                'task': 'classification',
                'model': 'Random Forest Classifier',
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'feature_importance': dict(zip(features, model.feature_importances_.tolist()))
            }

        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        results['cv_mean'] = float(cv_scores.mean())
        results['cv_std'] = float(cv_scores.std())

        return results

    @staticmethod
    def create_visualization(
        data: pd.DataFrame,
        plot_type: str,
        x: Optional[str] = None,
        y: Optional[str] = None,
        color: Optional[str] = None
    ) -> go.Figure:
        """
        Create interactive visualizations

        Args:
            data: Input dataframe
            plot_type: Type of plot (scatter, box, histogram, etc.)
            x: X-axis column
            y: Y-axis column
            color: Color grouping column

        Returns:
            Plotly figure
        """
        if plot_type == 'scatter':
            fig = px.scatter(data, x=x, y=y, color=color, trendline='ols')
        elif plot_type == 'box':
            fig = px.box(data, x=x, y=y, color=color)
        elif plot_type == 'histogram':
            fig = px.histogram(data, x=x, color=color, marginal='box')
        elif plot_type == 'correlation_heatmap':
            corr = data.select_dtypes(include=[np.number]).corr()
            fig = px.imshow(corr, text_auto=True, aspect='auto')
        elif plot_type == 'violin':
            fig = px.violin(data, x=x, y=y, color=color, box=True)
        else:
            fig = px.scatter(data, x=x, y=y)

        fig.update_layout(template='plotly_white', height=600)

        return fig


class ExperimentTemplates:
    """Pre-built experiment templates for common research scenarios"""

    @staticmethod
    def ab_test_template() -> Dict[str, Any]:
        """A/B testing experimental design"""
        return {
            'name': 'A/B Testing',
            'description': 'Compare two variants to determine which performs better',
            'methodology': 'Randomized controlled trial with two groups',
            'sample_size_calculation': 'Use power analysis with expected effect size',
            'variables': {
                'independent': ['Group (A/B)'],
                'dependent': ['Primary metric (e.g., conversion rate, engagement)'],
                'control': ['Time period', 'User characteristics']
            },
            'analysis_plan': [
                'Check randomization balance',
                'Test normality of dependent variable',
                'Perform t-test or Mann-Whitney U test',
                'Calculate effect size (Cohen\'s d)',
                'Report confidence intervals'
            ],
            'statistical_tests': ['t-test', 'chi-square', 'effect size'],
            'sample_code': """
# A/B Test Analysis
from paperagent.tools.advanced_stats import AdvancedStatistics

stats_tool = AdvancedStatistics()

# Compare groups
result = stats_tool.ttest_analysis(group_a_data, group_b_data)
print(f"P-value: {result['p_value']}")
print(f"Effect size: {result['cohens_d']}")
"""
        }

    @staticmethod
    def survey_analysis_template() -> Dict[str, Any]:
        """Survey data analysis template"""
        return {
            'name': 'Survey Analysis',
            'description': 'Analyze survey responses and relationships',
            'methodology': 'Cross-sectional survey with Likert scales',
            'variables': {
                'independent': ['Demographics', 'Categories'],
                'dependent': ['Survey responses', 'Satisfaction scores'],
                'control': ['Sample characteristics']
            },
            'analysis_plan': [
                'Descriptive statistics for all variables',
                'Reliability analysis (Cronbach\'s alpha)',
                'Correlation analysis between constructs',
                'Group comparisons (ANOVA)',
                'Factor analysis for scale validation'
            ],
            'statistical_tests': ['descriptive', 'correlation', 'ANOVA', 'reliability'],
            'sample_code': """
# Survey Analysis
stats = AdvancedStatistics()

# Descriptive statistics
desc = stats.descriptive_stats(survey_data)

# Correlation analysis
corr = stats.correlation_analysis(survey_data, method='spearman')

# Group comparison
anova = stats.anova_analysis(survey_data, 'satisfaction', 'age_group')
"""
        }

    @staticmethod
    def machine_learning_experiment_template() -> Dict[str, Any]:
        """Machine learning experiment template"""
        return {
            'name': 'Machine Learning Experiment',
            'description': 'Train and evaluate ML models',
            'methodology': 'Supervised learning with cross-validation',
            'variables': {
                'features': ['Input features'],
                'target': ['Prediction target'],
                'hyperparameters': ['Model parameters']
            },
            'analysis_plan': [
                'Exploratory data analysis',
                'Feature engineering and selection',
                'Train-test split',
                'Model training with cross-validation',
                'Hyperparameter tuning',
                'Model evaluation and comparison',
                'Feature importance analysis'
            ],
            'evaluation_metrics': ['accuracy', 'precision', 'recall', 'F1-score', 'AUC-ROC'],
            'sample_code': """
# ML Experiment
stats = AdvancedStatistics()

# Train model
result = stats.machine_learning_analysis(
    data=df,
    target='outcome',
    features=['feature1', 'feature2', 'feature3'],
    task='classification'
)

print(f"Accuracy: {result['accuracy']}")
print(f"Feature importance: {result['feature_importance']}")
"""
        }

    @staticmethod
    def get_all_templates() -> Dict[str, Dict[str, Any]]:
        """Get all available experiment templates"""
        return {
            'ab_test': ExperimentTemplates.ab_test_template(),
            'survey': ExperimentTemplates.survey_analysis_template(),
            'ml_experiment': ExperimentTemplates.machine_learning_experiment_template(),
            'correlation_study': ExperimentTemplates.correlation_study_template(),
            'longitudinal_study': ExperimentTemplates.longitudinal_study_template(),
            'quasi_experimental': ExperimentTemplates.quasi_experimental_template()
        }

    @staticmethod
    def correlation_study_template() -> Dict[str, Any]:
        """Correlation study template"""
        return {
            'name': 'Correlation Study',
            'description': 'Examine relationships between variables',
            'methodology': 'Correlational design',
            'analysis_plan': [
                'Scatterplot matrix',
                'Pearson/Spearman correlation',
                'Partial correlation controlling for confounds',
                'Multiple regression for prediction'
            ],
            'statistical_tests': ['correlation', 'regression']
        }

    @staticmethod
    def longitudinal_study_template() -> Dict[str, Any]:
        """Longitudinal study template"""
        return {
            'name': 'Longitudinal Study',
            'description': 'Track changes over time',
            'methodology': 'Repeated measures design',
            'analysis_plan': [
                'Repeated measures ANOVA',
                'Mixed-effects modeling',
                'Time series analysis',
                'Growth curve modeling'
            ],
            'statistical_tests': ['repeated measures ANOVA', 'mixed models']
        }

    @staticmethod
    def quasi_experimental_template() -> Dict[str, Any]:
        """Quasi-experimental design template"""
        return {
            'name': 'Quasi-Experimental Design',
            'description': 'Compare groups without random assignment',
            'methodology': 'Natural experiment or matched groups',
            'analysis_plan': [
                'Propensity score matching',
                'Difference-in-differences',
                'Regression discontinuity',
                'Covariate adjustment'
            ],
            'statistical_tests': ['ANCOVA', 'matching', 'DiD']
        }


# Export
__all__ = ['AdvancedStatistics', 'ExperimentTemplates']
