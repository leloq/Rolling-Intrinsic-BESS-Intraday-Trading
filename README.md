
# ⚡ Rolling Intrinsic Intraday Energy Trading Optimization

## Project Overview
The Rolling Intrinsic Intraday Energy Trading Optimization repository models continous intraday trading with the means of discretization and linear optimization. While the results of the underlying paper are based on actual EPEX Spot data (which requires paid access), the open-source repository at hand provides randomly generated data. 

## Project Structure

The notebook 'Create Randomized Intraday Transaction Data' constitutes the first step: random transaction data (continous buy/sell orders) are created. Then, in the folder 'Code/Rolling Intrinsic' the Python scripts 'Rolling Intrinsic H.py' and 'Rolling Intrinsic QH.py' can be executed to model rolling intrinsic trading, respectively on the market for hourly or quarter hourly products. The Python script executes the function simulate_period() with the parameters described below. Finally, the results are then saved in the 'results' folder.

## Parameter Descriptions

1. **threshold**: 
   - **Description**: This is the relative price change threshold as a percentage. It represents the minimum price change required to trigger an action (buy/sell).
   - **Example**: A threshold of 0 means that even the smallest price fluctuation can trigger trading decisions.

2. **threshold_abs_min**: 
   - **Description**: This is the absolute minimum price change threshold. It ensures that there’s a fixed minimum price movement (in absolute terms) that must occur before any trading action is triggered, regardless of percentage changes.
   - **Example**: A value of 0 means no minimum price movement is required.

3. **discount_rate**: 
   - **Description**: This is used for time-based price discounting. It represents the percentage rate at which the future value of cash flows is discounted.
   - **Example**: A discount rate of 0 means no time discounting is applied to prices, treating future prices as if they were happening now.

4. **bucket_size**: 
   - **Description**: This refers to the time interval (in minutes) for grouping or aggregating trades.
   - **Example**: For example, with a bucket size of 15, trades are grouped into 15-minute intervals, helping to control the granularity of the optimization process.

5. **c_rate**: 
   - **Description**: This is the charge rate for the battery storage system. It defines how fast the system can charge or discharge as a fraction of the total storage capacity.
   - **Example**: A value of 0.5 means the battery can charge or discharge up to 50% of its capacity in one unit of time.

6. **roundtrip_eff**: 
   - **Description**: This refers to the roundtrip efficiency of the battery storage system. It represents the overall efficiency of the battery when charging and discharging.
   - **Example**: A roundtrip efficiency of 0.86 means that for every 100 units of energy stored, 86 units are available for use after charging and discharging.

7. **max_cycles**: 
   - **Description**: This defines the maximum number of charge-discharge cycles the battery can perform in a given period (e.g., a year).
   - **Example**: A value of 365 means the battery can undergo one full cycle (charge + discharge) per day on average, limiting the total wear on the battery system.


## 🚀 Getting Started

### Step 1: PostgreSQL Database Setup
First, you need to set up the PostgreSQL database. You can follow these steps:

Open a terminal and run the PostgreSQL command-line interface:

```bash
psql -d postgres
```

#### Create the `intradaydb` Database
Create the database for your project:

```sql
CREATE DATABASE intradaydb;
```

#### Create the `leloq` User
Create a user named `leloq` with the password `123`:

```sql
CREATE USER leloq WITH PASSWORD '123';
```

#### Grant Privileges to the `leloq` User
Grant all privileges on the `intradaydb` database to the `leloq` user:

```sql
GRANT ALL PRIVILEGES ON DATABASE intradaydb TO leloq;
```

#### Exit the PostgreSQL Shell
Type `\q` to exit the PostgreSQL shell.

```bash
\q
```

### Step 2: Python Environment Setup
Next, set up the Python environment. You'll need Python 3.8 or higher. You can set up a virtual environment and install the required dependencies as follows:

#### Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

#### Install the required dependencies:
Install all required libraries listed in the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

#### Install Gurobi:
Gurobi is used as the solver in this project. Make sure you have a valid Gurobi license. If you haven't installed it yet, you can follow these steps:

Install Gurobi Python bindings:

```bash
pip install gurobipy
```

Set up your Gurobi license (follow the instructions on the Gurobi website).

### Step 3: Create randomized transaction data (or use actual EPEX Spot data)

Now, you can run the code from the notebook 'Create Randomized Intraday Transaction Data'. Alternatively, you can fill the intradaydb table with actual data from any continous intraday market you want to cover.

### Step 4: Running the Code
Now you're ready to run the optimization code in the "Code/Rolling Intrinsic" folder.


## ⚙️ Dependencies
- **Python**: 3.8+
- **PostgreSQL**: 13+
- **Gurobi**: Optimization solver
- **Libraries**:
  - `psycopg2`
  - `pandas`
  - `numpy`
  - `gurobipy`
  - `sqlalchemy`
  - `pulp`

## 📄 License
This project is licensed under the MIT License - see the `LICENSE` file for details.

## 🛠️ Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## 👤 Contributors
Leo Semmelmann, Jannik Dresselhaus, Kim Miskiw, Jan Niklas Ludwig


