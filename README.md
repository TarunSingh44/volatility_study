# Volatility Study

## Table of Content
- [Important Contacts](#contacts)
- [Documentation](#documentation)
- [Instruction for setting up the repo](#setup)
- [As a Contributor](#contrib)

# Important Contacts: <a name="contacts"></a>
- [Tarun Singh] - [singht1@tcd.ie]
- [Darren Scott] - [DASCOTT@tcd.ie]
- [Khurshid Ahmad] - [Khurshid.Ahmad@tcd.ie]

# Documentation <a name="documentation"></a>

## Project Summary

### Book for reference 
[Asset Price Dynamics, Volatility, and Prediction](https://github.com/TarunSingh44/volatility_study/blob/main/docs/Asset%20Price%20Dynamics%2C%20Volatility%2C%20and%20Prediction.pdf)

### Base code for reference
[Base Code Developed based on the book](https://github.com/TarunSingh44/volatility_study/blob/main/exploratory_notebooks/Volatility_Backend_Full_Code.ipynb)

### Volatility Study Folder Structure 
```
<BASE_PATH>/
│
└── data/
    ├── raw/
    │   ├── raw_data/
    │   └── raw_stats/
    ├── cleaned/
    │   ├── cleaned_data/
    │   └── cleaned_stats/
    └── volatility_stats/
        ├── statistical_moments/
        │   ├── actual/
        │   └── return/
        ├── correlation/
        │   ├── actual/
        │   └── return/
        ├── rolling_stats/
        │   ├── actual/
        │   │   ├── data/
        │   │   │   ├── rolling_mean/
        │   │   │   └── rolling_std/
        │   │   └── plots/
        │   │       ├── rolling_mean/
        │   │       └── rolling_std/
        │   └── return/
        │       ├── data/
        │       │   ├── rolling_mean_return/
        │       │   └── rolling_std_return/
        │       └── plots/
        │           ├── rolling_mean_return/
        │           └── rolling_std_return/
        ├── abs_squared_return/
        │   ├── data/
        │   │   ├── abs_return/
        │   │   └── squared_return/
        │   └── plots/
        │       ├── abs_return/
        │       └── squared_return/
        ├── log_return/
        │   ├── data/
        │   └── plots/
        ├── cross_auto_correlation/
        │   └── data/
        ├── q_statistic/
        │   └── data/
        └── variance_ratio_test/
            └── data/
```

# Instruction for setting up the repo <a name="setup"></a>

1. Clone the repo
   `git clone https://github.com/TarunSingh44/volatility_study.git`
   
2. Create a virtual env and install the dependencies
   `pip install -r requirements.txt`
   
3. Run the python script
   `python run.py`

# As a Contributor <a name="contrib"></a>

This is a developer contribution guide for students under Prof Khurshid wanting to contribute to the Volatility Study project. 

## Getting Started

>In order to contribute to this project, you must create a new branch off the main branch.

Install the following on your local environment

* Python > 3.10 =< 3.12

### Adding Code Contributions 

### Opening a PR

Once you have your contribution implemented and tested, open a pull request back to the main branch being used. Once the PR is opened, you need to assign Prof Khurshid or Darren Scott as reviewers.

You should also leave a detailed comment in the PR with the following information:

- The updates that are included in this PR 
- Any end-to-end testing performed and the results of that testing

### Example comment

```
*[Description]*

- Detailed breakdown of the task

*[Test Results]*

- Include the tests you performed
```

### Merging the PR and final Release

Once the PR is ready to merge, you should delete the branch you were working on to keep the GIT branches clean. 



