import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(filename='process_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def load_data(file_path):
    """Load CSV file into a DataFrame."""
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Successfully loaded {file_path}")
        return df
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        raise


def check_data_quality(df, file_name):
    """Check for missing values, data types, unique values, outliers, and invalid formats."""
    quality_report = []

    # Missing Values
    for column in df.columns:
        missing_values = df[column].isnull().sum()
        data_type = df[column].dtype

        # Unique Values
        unique_values = df[column].nunique()
        top_value = df[column].mode()[0] if not df[column].empty else None
        frequency = df[column].value_counts(
        ).iloc[0] if not df[column].empty else 0

        # Outliers (for numeric columns)
        outliers = None
        if pd.api.types.is_numeric_dtype(df[column]):
            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = ((df[column] < lower_bound) |
                        (df[column] > upper_bound)).sum()

        # Invalid Formats (for specific columns)
        invalid_format = None
        if column == 'email':
            invalid_format = df[column].apply(
                lambda x: '@' not in str(x)).sum()
        elif column == 'phone':
            invalid_format = df[column].apply(
                lambda x: not str(x).isdigit()).sum()

        # Add to report
        quality_report.append({
            'Column': column,
            'Missing Values': missing_values,
            'Data Type': data_type,
            'Unique Values': unique_values,
            'Top Value': top_value,
            'Frequency': frequency,
            'Outliers': outliers,
            'Invalid Format': invalid_format
        })

    # Duplicate Rows
    duplicate_rows = df.duplicated().sum()

    # Generate Report
    report_df = pd.DataFrame(quality_report)
    report_df.to_csv(
        f'data_quality_report_{file_name}.txt', index=False, sep='\t')
    logging.info(f"Generated data quality report for {file_name}")

    # Log Summary
    logging.info(f"Summary for {file_name}:")
    logging.info(f"  Missing Values: {report_df['Missing Values'].sum()}")
    logging.info(f"  Duplicate Rows: {duplicate_rows}")
    logging.info(
        f"  Columns with Outliers: {report_df[report_df['Outliers'] > 0]['Column'].tolist()}")
    logging.info(
        f"  Columns with Invalid Formats: {report_df[report_df['Invalid Format'] > 0]['Column'].tolist()}")


def handle_missing_values(df):
    """Handle missing values by filling or dropping."""
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].fillna(df[column].mean())
        else:
            df[column] = df[column].fillna("Unknown")
    logging.info("Handled missing values")
    return df


def main():
    try:
        # Step 1: Data Loading & Validation
        logging.info("Starting E-commerce Data Processing")

        # Load data
        transactions = load_data('transactions.csv')
        products = load_data('products.csv')
        customers = load_data('customers.csv')

        # Check data quality
        check_data_quality(transactions, 'transactions')
        check_data_quality(products, 'products')
        check_data_quality(customers, 'customers')

        # Handle missing values
        transactions = handle_missing_values(transactions)
        products = handle_missing_values(products)
        customers = handle_missing_values(customers)

        logging.info("E-commerce Data Processing Completed Successfully")

    except Exception as e:
        logging.error(f"An error occurred during processing: {e}")
        with open('error_report.txt', 'w') as f:
            f.write(f"Error: {e}")


if __name__ == "__main__":
    main()
