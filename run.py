import yaml
import logging
from src.config.constants import(
    LOGGING_FILE,
    RAW_DATA,
    GDRIVE_ID,
    CLEANED_DATA,
    STATISTICAL_MOMENTS_ACTUAL,
    STATISTICAL_MOMENTS_RETURN,
    CORRELATION_ACTUAL,
    CORRELATION_RETURN,
    STATISTICAL_MOMENTS_SQUARED_RETURN,
    STATISTICAL_MOMENTS_ABSOLUTE_RETURN,
    STATISTICAL_MOMENTS_LOG_ABS_R_MINUS_R_BAR_RETURN,
    FREQUENCIES,
)
from src.acquire_data.download_dataset import download_and_unzip_from_gdrive
from src.data_preprocessing.raw_data_stats import generate_raw_data_stats
from src.data_preprocessing.clean_raw_data import raw_data_cleaning
from src.statistical_analysis.statistical_moments import calc_stats_moments
from src.statistical_analysis.correlation import calc_correlation
from src.statistical_analysis.rolling_mean_sd import calc_rolling_mean_sd
from src.statistical_analysis.abs_return_sq_return import calc_abs_squared_return_stats
from src.statistical_analysis.log_return import calc_log_return_stats
from src.statistical_analysis.auto_corr import calc_cross_auto_corr_stats
from src.statistical_analysis.q_stats import calc_q_statistic_stats
from src.statistical_analysis.z_return import calc_vr_statistic_stats
from src.statistical_analysis.statistical_moments_output import output_stats_moments

with open(LOGGING_FILE, 'r') as f:
    config = yaml.safe_load(f)
    logging.config.dictConfig(config)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    
    #########################################################
    ## STEP 1: Acquire Raw Data Files from Google Drive #####
    #########################################################

    logger.info("Step 1: Starting to download data from Google Drive")
    download_and_unzip_from_gdrive(GDRIVE_ID, RAW_DATA)
    logger.info("Step 1: All raw data files acquired successfully.")

    ############################################################
    ##### STEP 2: Get Raw Data Stats                       #####
    ############################################################

    logger.info("Step 2: Generating Raw Data Stats")
    generate_raw_data_stats()
    logger.info("Step 2: Raw Data Stats Generated.")

    ############################################################
    ##### STEP 3: Clean Data & Save Statistics             #####
    ############################################################

    logger.info("Step 3: Starting Cleaning Process")
    raw_data_cleaning()
    logger.info("Step 3: Data Cleaning Completed.")

    ############################################################
    ##### STEP 4: Volatility Stats                         #####
    ############################################################

    # Statistical Moments
    logger.info("Step 4.1: Calculating Statistical Moments")
    calc_stats_moments(CLEANED_DATA, STATISTICAL_MOMENTS_ACTUAL, STATISTICAL_MOMENTS_RETURN)
    logger.info("Step 4.1: Statistical Moments Completed.")

    # Correlation
    logger.info("Step 4.2: Calculating Correlation Matrix")
    calc_correlation(CLEANED_DATA, CORRELATION_ACTUAL, CORRELATION_RETURN)
    logger.info("Step 4.2: Correlation Matrix Completed.")

    # Rolling Mean & SD
    logger.info("Step 4.3: Calculating Rolling Mean & SD")
    calc_rolling_mean_sd(CLEANED_DATA, plot=False)
    logger.info("Step 4.3: Rolling Mean & SD Completed.")

    # Abs Return & Sq Return
    logger.info("Step 4.4: Calculating Abs Return & Sq Return")
    calc_abs_squared_return_stats(CLEANED_DATA, plot=False)
    logger.info("Step 4.4: Abs Return & Sq Return Completed.")

    # Log Return
    logger.info("Step 4.5: Calculating Log Return")
    calc_log_return_stats(CLEANED_DATA, plot=False)
    logger.info("Step 4.5: Log Return Completed.")

    # Sample Autocorrelation 
    logger.info("Step 4.6: Calculating Sample Autocorrelation")
    calc_cross_auto_corr_stats(CLEANED_DATA)
    logger.info("Step 4.6: Sample Autocorrelation Completed.")

    # Q-Stats
    logger.info("Step 4.7: Calculating Q-Stats")
    calc_q_statistic_stats(CLEANED_DATA, max_tau=30, n_jobs=4)
    logger.info("Step 4.7: Q-Stats Completed.")

    # Z Return
    logger.info("Step 4.8: Calculating VR test")
    calc_vr_statistic_stats(CLEANED_DATA, n_jobs=4)
    logger.info("Step 4.8: VR test Completed.")

    # Statistical Moments for Actual, Return, Abs Return, Sq Return and Log(abs( r - avg[ r ] ) )  (All Frequencies)
    logger.info("Step 4.9: Calculating Statistical Moments for Actual, Return, Abs Return, Sq Return & Log Return")
    output_stats_moments(
        CLEANED_DATA,
        STATISTICAL_MOMENTS_ACTUAL,
        STATISTICAL_MOMENTS_RETURN,
        STATISTICAL_MOMENTS_SQUARED_RETURN,
        STATISTICAL_MOMENTS_ABSOLUTE_RETURN,
        STATISTICAL_MOMENTS_LOG_ABS_R_MINUS_R_BAR_RETURN,
        FREQUENCIES,
    )
    logger.info("Step 4.9: Statistical Moments Completed.")
