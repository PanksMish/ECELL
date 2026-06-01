"""
EOSNM Data Loader and Preprocessing Module

This module handles:
1. Loading raw E-Cell participation data
2. Feature engineering (skill scores, network metrics, barrier indices)
3. Data preprocessing and normalization
4. Train/validation/test splitting

Dataset format:
- student_id, department, year, skill_*, network_*, participation_*, placement_outcome
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import logging

from eosnm_core import StudentState

logger = logging.getLogger(__name__)


class ECellDataLoader:
    """
    Loads and preprocesses E-Cell student participation data.
    """
    
    def __init__(self, data_path: str = None, random_state: int = 42):
        """
        Initialize data loader.
        
        Args:
            data_path: Path to CSV data file (if None, generates synthetic data)
            random_state: Random seed for reproducibility
        """
        self.data_path = data_path
        self.random_state = random_state
        self.scaler_continuous = StandardScaler()
        self.scaler_counts = MinMaxScaler()
        
        logger.info(f"ECellDataLoader initialized with random_state={random_state}")
    
    def generate_synthetic_data(self, n_students: int = 391) -> pd.DataFrame:
        """
        Generate synthetic E-Cell dataset matching paper specifications.
        
        Based on the paper's description:
        - 391 undergraduate engineering students
        - Multi-dimensional skill scores
        - Professional network degrees
        - Participation metrics
        - Placement outcomes
        
        Args:
            n_students: Number of students (default: 391 from paper)
            
        Returns:
            DataFrame with student records
        """
        np.random.seed(self.random_state)
        
        logger.info(f"Generating synthetic dataset with {n_students} students...")
        
        # Demographics
        departments = ['CS', 'IT', 'Electronics', 'Mechanical', 'Civil']
        years = [2, 3, 4]  # Sophomore, Junior, Senior
        
        data = {
            'student_id': range(1, n_students + 1),
            'department': np.random.choice(departments, n_students),
            'year_of_study': np.random.choice(years, n_students),
            
            # Skill components (normalized to 0-1, then scaled)
            'communication_skill': np.random.beta(2, 2, n_students),
            'problem_solving': np.random.beta(2.5, 2, n_students),
            'leadership': np.random.beta(2, 2.5, n_students),
            'teamwork': np.random.beta(2.5, 2, n_students),
            'digital_skills': np.random.beta(2, 2, n_students),
            
            # Participation metrics
            'workshop_attended': np.random.poisson(3, n_students),
            'competitions_participated': np.random.poisson(2, n_students),
            'mentoring_sessions': np.random.poisson(4, n_students),
            'networking_events': np.random.poisson(2, n_students),
            
            # Network metrics
            'peer_connections': np.random.poisson(8, n_students),
            'mentor_connections': np.random.poisson(2, n_students),
            'industry_connections': np.random.poisson(1, n_students),
            
            # Barrier components (higher = more barriers)
            'time_constraint': np.random.beta(3, 2, n_students),
            'scheduling_conflict': np.random.beta(2.5, 2.5, n_students),
            'awareness_gap': np.random.beta(2, 3, n_students),
            'resource_limitation': np.random.beta(2, 3, n_students),
        }
        
        df = pd.DataFrame(data)
        
        # Compute composite features
        df['baseline_skill_score'] = (
            df['communication_skill'] * 0.2 +
            df['problem_solving'] * 0.25 +
            df['leadership'] * 0.2 +
            df['teamwork'] * 0.2 +
            df['digital_skills'] * 0.15
        ) * 10  # Scale to 0-10
        
        df['network_degree'] = (
            df['peer_connections'] +
            df['mentor_connections'] * 2 +  # Weight mentors more
            df['industry_connections'] * 3   # Weight industry most
        )
        
        df['participation_intensity'] = (
            df['workshop_attended'] * 0.3 +
            df['competitions_participated'] * 0.3 +
            df['mentoring_sessions'] * 0.2 +
            df['networking_events'] * 0.2
        )
        
        # Normalize participation intensity to 0-1
        df['participation_intensity'] = (
            df['participation_intensity'] / df['participation_intensity'].max()
        )
        
        df['barrier_index'] = (
            df['time_constraint'] * 0.3 +
            df['scheduling_conflict'] * 0.25 +
            df['awareness_gap'] * 0.25 +
            df['resource_limitation'] * 0.2
        )
        
        # Generate placement outcomes based on employability factors
        # Higher skill + network + lower barriers → higher placement probability
        employability_score = (
            0.5 * (df['baseline_skill_score'] / 10) +
            0.3 * (df['network_degree'] / df['network_degree'].max()) +
            0.2 * (1 - df['barrier_index'])
        )
        
        # Add some noise and apply logistic function
        placement_prob = 1 / (1 + np.exp(-5 * (employability_score - 0.5)))
        df['placement_outcome'] = (np.random.random(n_students) < placement_prob).astype(int)
        
        # Add some prior entrepreneurship exposure
        df['prior_entrepreneurship'] = np.random.binomial(1, 0.3, n_students)
        
        logger.info(f"Synthetic data generated: {len(df)} students, "
                   f"{df['placement_outcome'].sum()} placements "
                   f"({df['placement_outcome'].mean()*100:.1f}%)")
        
        return df
    
    def load_data(self) -> pd.DataFrame:
        """
        Load data from file or generate synthetic data.
        
        Returns:
            DataFrame with student records
        """
        if self.data_path is not None:
            logger.info(f"Loading data from {self.data_path}")
            df = pd.read_csv(self.data_path)
        else:
            logger.info("No data path provided, generating synthetic data")
            df = self.generate_synthetic_data()
        
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data: handle missing values, normalize, encode categoricals.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Preprocessing data...")
        
        df = df.copy()
        
        # Handle missing values
        continuous_cols = ['baseline_skill_score', 'participation_intensity']
        for col in continuous_cols:
            if col in df.columns and df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        # Handle categorical missing values
        categorical_cols = ['department', 'year_of_study']
        for col in categorical_cols:
            if col in df.columns and df[col].isnull().any():
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        # One-hot encode department
        if 'department' in df.columns:
            df = pd.get_dummies(df, columns=['department'], prefix='dept')
        
        # Clip outliers (3 standard deviations)
        for col in ['baseline_skill_score', 'network_degree', 'participation_intensity']:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                df[col] = df[col].clip(mean - 3*std, mean + 3*std)
        
        logger.info("Preprocessing complete")
        
        return df
    
    def create_student_states(self, df: pd.DataFrame) -> List[StudentState]:
        """
        Convert DataFrame rows to StudentState objects.
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            List of StudentState objects
        """
        students = []
        
        for _, row in df.iterrows():
            # Extract demographic features (one-hot encoded departments)
            dept_cols = [col for col in df.columns if col.startswith('dept_')]
            demographics = {
                'year': row['year_of_study'],
                'prior_experience': row.get('prior_entrepreneurship', 0)
            }
            
            # Add department info
            for col in dept_cols:
                demographics[col] = row[col]
            
            student = StudentState(
                student_id=int(row['student_id']),
                skill_level=float(row['baseline_skill_score']),
                network_degree=int(row['network_degree']),
                barrier_index=float(row['barrier_index']),
                participation_intensity=float(row['participation_intensity']),
                demographics=demographics
            )
            
            students.append(student)
        
        logger.info(f"Created {len(students)} StudentState objects")
        
        return students
    
    def split_data(self, df: pd.DataFrame, 
                   test_size: float = 0.2, 
                   val_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/validation/test sets.
        
        Args:
            df: Full DataFrame
            test_size: Proportion for test set
            val_size: Proportion for validation set
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=df['placement_outcome']
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio,
            random_state=self.random_state,
            stratify=train_val_df['placement_outcome']
        )
        
        logger.info(f"Data split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        
        return train_df, val_df, test_df


def load_ecell_data(data_path: str = None, 
                    preprocess: bool = True,
                    return_states: bool = True) -> Tuple:
    """
    Convenience function to load and prepare E-Cell data.
    
    Args:
        data_path: Path to CSV file (None for synthetic data)
        preprocess: Whether to preprocess data
        return_states: Whether to return StudentState objects
        
    Returns:
        If return_states=True: (students_list, df)
        If return_states=False: df
    """
    loader = ECellDataLoader(data_path=data_path)
    df = loader.load_data()
    
    if preprocess:
        df = loader.preprocess_data(df)
    
    if return_states:
        students = loader.create_student_states(df)
        return students, df
    
    return df


def save_synthetic_dataset(output_path: str = 'data/ecell_data.csv', 
                          n_students: int = 391):
    """
    Generate and save synthetic dataset to file.
    
    Args:
        output_path: Where to save CSV
        n_students: Number of students
    """
    import os
    
    # Create directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    loader = ECellDataLoader()
    df = loader.generate_synthetic_data(n_students)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Synthetic dataset saved to {output_path}")
    
    return output_path


if __name__ == "__main__":
    # Test data generation
    logging.basicConfig(level=logging.INFO)
    
    # Generate and save synthetic dataset
    output_file = save_synthetic_dataset('data/ecell_data.csv', n_students=391)
    
    # Load and verify
    students, df = load_ecell_data(data_path=output_file)
    
    print(f"\nDataset Summary:")
    print(f"Total students: {len(students)}")
    print(f"\nDataFrame shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nPlacement rate: {df['placement_outcome'].mean()*100:.1f}%")
