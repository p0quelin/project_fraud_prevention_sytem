# Functions from generator.ipynb
import os
import random
import numpy as np
import pandas as pd
import datetime
import time
from pandarallel import pandarallel

# Initialize pandarallel at import time
pandarallel.initialize(progress_bar=False, nb_workers=4)

# Configuration
FRAUD_CONFIG = {
    'target_fraud_rate': 0.002,  # 0.2% overall fraud rate (industry average)
    'threshold_multiplier': 3,    # Number of standard deviations for anomaly detection
    'scenario_distribution': {
        1: 0.15,  # Anomalous amount fraud (15% of frauds)
        2: 0.35,  # Compromised terminal (35% of frauds)
        3: 0.30,  # Card-not-present fraud (30% of frauds)
        4: 0.20   # Quick cash-out fraud (20% of frauds)
    },
    'fraud_probabilities': {
        'anomalous_amount': 0.001,     # Base probability for anomalous amounts
        'compromised_terminal': 0.002,  # Base probability for compromised terminals
        'cnp_fraud': 0.0015,           # Base probability for card-not-present fraud
        'quick_cashout': 0.001         # Base probability for quick cash-out fraud
    }
}

def add_seasonal_patterns(transactions_df, start_date):
    """Add seasonal patterns to transaction amounts based on time of day and week"""
    # Convert TX_DATETIME to pandas datetime if it's not already
    if not isinstance(transactions_df.TX_DATETIME.iloc[0], pd.Timestamp):
        transactions_df['TX_DATETIME'] = pd.to_datetime(transactions_df.TX_DATETIME)
    
    # Add hour of day
    transactions_df['hour'] = transactions_df.TX_DATETIME.dt.hour
    transactions_df['month'] = transactions_df.TX_DATETIME.dt.month
    transactions_df['day_of_week'] = transactions_df.TX_DATETIME.dt.dayofweek
    
    # Define time periods
    def get_time_period(hour):
        if hour < 6:
            return 'late_night'
        elif hour < 12:
            return 'morning'
        elif hour < 18:
            return 'afternoon'
        else:
            return 'evening'
    
    transactions_df['time_period'] = transactions_df.hour.apply(get_time_period)
    
    # Apply seasonal patterns
    conditions = [
        # Late night (midnight - 6am): Lower amounts, fewer transactions
        transactions_df.time_period == 'late_night',
        # Morning rush (6am - noon): Medium amounts, many transactions
        transactions_df.time_period == 'morning',
        # Afternoon (noon - 6pm): Highest amounts, most transactions
        transactions_df.time_period == 'afternoon',
        # Evening (6pm - midnight): Medium-high amounts
        transactions_df.time_period == 'evening'
    ]
    
    multipliers = [0.7, 1.0, 1.2, 1.1]
    
    # Apply time-of-day adjustments
    transactions_df['TX_AMOUNT'] = transactions_df['TX_AMOUNT'] * np.select(conditions, multipliers)
    
    # Weekend adjustments
    is_weekend = transactions_df.day_of_week.isin([5, 6])  # Saturday or Sunday
    transactions_df.loc[is_weekend, 'TX_AMOUNT'] *= 1.1  # 10% higher amounts on weekends
    
    # Monthly patterns (e.g., higher spending around payday)
    is_month_end = transactions_df.TX_DATETIME.dt.day >= 25
    transactions_df.loc[is_month_end, 'TX_AMOUNT'] *= 1.15  # 15% higher at month end
    
    # Round amounts to 2 decimal places
    transactions_df['TX_AMOUNT'] = transactions_df['TX_AMOUNT'].round(2)
    
    # Ensure no negative amounts
    transactions_df.loc[transactions_df.TX_AMOUNT < 0, 'TX_AMOUNT'] = \
        transactions_df.loc[transactions_df.TX_AMOUNT < 0, 'TX_AMOUNT'].abs()
    
    return transactions_df

def generate_customer_profiles_table(n_customers, random_state=0):
    np.random.seed(random_state)
        
    customer_id_properties = []
    
    # Customer segments with their properties
    segments = {
        'low_value': {
            'prop': 0.3,  # 30% of customers
            'amount_range': (5, 50),
            'tx_per_day': (0.1, 1),
            'std_factor': 0.3
        },
        'medium_value': {
            'prop': 0.5,  # 50% of customers
            'amount_range': (20, 100),
            'tx_per_day': (0.5, 2),
            'std_factor': 0.5
        },
        'high_value': {
            'prop': 0.2,  # 20% of customers
            'amount_range': (50, 500),
            'tx_per_day': (1, 4),
            'std_factor': 0.7
        }
    }
    
    for customer_id in range(n_customers):
        # Assign customer to segment
        rand_val = np.random.random()
        cum_prop = 0
        for segment, props in segments.items():
            cum_prop += props['prop']
            if rand_val <= cum_prop:
                segment_props = props
                break
                
        # Location uniformly distributed in 100x100 grid
        x_customer_id = np.random.uniform(0, 100)
        y_customer_id = np.random.uniform(0, 100)
        
        # Generate customer properties based on segment
        mean_amount = np.random.uniform(
            segment_props['amount_range'][0],
            segment_props['amount_range'][1]
        )
        
        # Standard deviation varies by segment
        std_amount = mean_amount * segment_props['std_factor']
        
        # Transactions per day varies by segment
        mean_nb_tx_per_day = np.random.uniform(
            segment_props['tx_per_day'][0],
            segment_props['tx_per_day'][1]
        )
        
        customer_id_properties.append([
            customer_id,
            x_customer_id, y_customer_id,
            mean_amount, std_amount,
            mean_nb_tx_per_day
        ])
        
    customer_profiles_table = pd.DataFrame(
        customer_id_properties, 
        columns=['CUSTOMER_ID', 'x_customer_id', 'y_customer_id',
                'mean_amount', 'std_amount', 'mean_nb_tx_per_day']
    )
    
    return customer_profiles_table

def generate_terminal_profiles_table(n_terminals, random_state=0):
    np.random.seed(random_state)
    
    # Define terminal types and their distribution
    terminal_types = {
        'retail': {
            'weight': 0.4,  # 40% retail terminals
            'clustering_factor': 0.7,  # tends to be in shopping areas
        },
        'atm': {
            'weight': 0.3,  # 30% ATMs
            'clustering_factor': 0.5,  # moderately clustered
        },
        'online': {
            'weight': 0.2,  # 20% online terminals
            'clustering_factor': 0.1,  # mostly random locations
        },
        'pos': {
            'weight': 0.1,  # 10% small point-of-sale
            'clustering_factor': 0.8,  # highly clustered
        }
    }
    
    terminal_profiles = []
    
    for terminal_id in range(n_terminals):
        # Determine terminal type
        terminal_type = np.random.choice(
            list(terminal_types.keys()),
            p=[t['weight'] for t in terminal_types.values()]
        )
        
        # Generate location with clustering based on terminal type
        if np.random.random() < terminal_types[terminal_type]['clustering_factor']:
            # Clustered location
            center_x = np.random.uniform(0, 100)
            center_y = np.random.uniform(0, 100)
            radius = np.random.uniform(1, 10)
            angle = np.random.uniform(0, 2 * np.pi)
            
            x_terminal_id = center_x + radius * np.cos(angle)
            y_terminal_id = center_y + radius * np.sin(angle)
            
            # Ensure coordinates are within bounds
            x_terminal_id = np.clip(x_terminal_id, 0, 100)
            y_terminal_id = np.clip(y_terminal_id, 0, 100)
        else:
            # Random location
            x_terminal_id = np.random.uniform(0, 100)
            y_terminal_id = np.random.uniform(0, 100)
        
        terminal_profiles.append([
            terminal_id,
            x_terminal_id,
            y_terminal_id,
            terminal_type
        ])
    
    terminal_profiles_table = pd.DataFrame(
        terminal_profiles,
        columns=['TERMINAL_ID', 'x_terminal_id', 'y_terminal_id', 'terminal_type']
    )
    
    return terminal_profiles_table

def get_list_terminals_within_radius(customer_profile, x_y_terminals, r):
    """
    Find terminals within radius r of customer location.
    Returns a list of terminal indices that are accessible to the customer.
    """
    try:
        # Location (x,y) of customer as numpy array
        x_y_customer = customer_profile[['x_customer_id','y_customer_id']].values.astype(float)
        
        # Squared difference in coordinates between customer and terminal locations
        squared_diff_x_y = np.square(x_y_customer - x_y_terminals)
        
        # Sum along rows and compute squared root to get distance
        dist_x_y = np.sqrt(np.sum(squared_diff_x_y, axis=1))
        
        # Get the indices of terminals which are at a distance less than r
        available_terminals = list(np.where(dist_x_y<r)[0])
        
        # If no terminals are in range, return the closest one
        if len(available_terminals)==0:
            available_terminals = [np.argmin(dist_x_y)]
            
        return available_terminals
        
    except Exception as e:
        print(f"Error processing customer {customer_profile.CUSTOMER_ID}: {str(e)}")
        # Return at least one random terminal as fallback
        return [np.random.randint(len(x_y_terminals))]

def generate_transactions_table(customer_profile, start_date="2024-01-01", nb_days=10):
    customer_transactions = []
    
    random.seed(customer_profile.CUSTOMER_ID)
    np.random.seed(customer_profile.CUSTOMER_ID)
    
    # For all days
    for day in range(nb_days):
        # Random number of transactions for that day 
        nb_tx = np.random.poisson(customer_profile.mean_nb_tx_per_day)
        
        # If nb_tx positive, let us generate transactions
        if nb_tx>0:
            for tx in range(nb_tx):
                # Time of transaction: Around noon, with a few hours of noise
                time_tx = int(np.random.normal(86400/2, 20000))
                # If transaction time between 0 and 24*60^2
                if (time_tx>0) and (time_tx<86400):
                    
                    # Amount is drawn from a normal distribution  
                    amount = np.random.normal(customer_profile.mean_amount, customer_profile.std_amount)
                    # If amount negative, draw from a uniform distribution
                    if amount<0:
                        amount = np.random.uniform(0,customer_profile.mean_amount*2)
                    
                    # Round amount to 2 decimals
                    amount = round(amount,2)
                    
                    if len(customer_profile.available_terminals)>0:
                        terminal_id = random.choice(customer_profile.available_terminals)
                    
                        customer_transactions.append([time_tx+day*86400, day,
                                                  customer_profile.CUSTOMER_ID, 
                                                  terminal_id, amount])
            
    customer_transactions = pd.DataFrame(customer_transactions, columns=['TX_TIME_SECONDS', 'TX_TIME_DAYS', 'CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT'])
    
    if len(customer_transactions) > 0:
        customer_transactions['TX_DATETIME'] = pd.to_datetime(customer_transactions["TX_TIME_SECONDS"], unit='s', origin=start_date)
        customer_transactions = customer_transactions[['TX_DATETIME','CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT','TX_TIME_SECONDS', 'TX_TIME_DAYS']]
    
    return customer_transactions

def add_frauds(customer_profiles_table, terminal_profiles_table, transactions_df):
    # By default, all transactions are genuine
    transactions_df['TX_FRAUD'] = 0
    transactions_df['TX_FRAUD_SCENARIO'] = 0
    
    # Scenario 1: Anomalous Transaction Amount (based on customer segment)
    for customer_id in customer_profiles_table.CUSTOMER_ID:
        customer_txs = transactions_df[transactions_df.CUSTOMER_ID == customer_id]
        if len(customer_txs) > 0:
            # Calculate customer's normal behavior
            tx_stats = customer_txs.groupby(pd.Grouper(key='TX_DATETIME', freq='D'))['TX_AMOUNT'].agg(['mean', 'std']).fillna(0)
            daily_mean = tx_stats['mean'].mean()
            daily_std = tx_stats['std'].mean()
            
            # Flag transactions that deviate significantly from customer's pattern
            threshold = daily_mean + (FRAUD_CONFIG['threshold_multiplier'] * daily_std)
            potential_frauds = (transactions_df.CUSTOMER_ID == customer_id) & (transactions_df.TX_AMOUNT > threshold)
            
            # Apply fraud probability
            fraud_mask = potential_frauds & (np.random.random(len(transactions_df)) < FRAUD_CONFIG['fraud_probabilities']['anomalous_amount'])
            transactions_df.loc[fraud_mask, 'TX_FRAUD'] = 1
            transactions_df.loc[fraud_mask, 'TX_FRAUD_SCENARIO'] = 1
    
    nb_frauds_scenario_1 = transactions_df.TX_FRAUD.sum()
    print("Number of frauds from scenario 1 (Anomalous Amount): "+str(nb_frauds_scenario_1))
    
    # Scenario 2: Compromised Terminal (focusing on specific terminal types)
    for day in range(transactions_df.TX_TIME_DAYS.max()):
        # ATMs are more likely to be compromised on weekends
        is_weekend = (day % 7) >= 5
        terminal_type_weights = {
            'atm': 0.6 if is_weekend else 0.4,    # ATMs most vulnerable
            'retail': 0.3 if is_weekend else 0.4,  # Retail terminals next
            'pos': 0.1,                           # Small POS less vulnerable
            'online': 0.0                         # Online terminals not affected
        }
        
        # Select potentially compromised terminals based on weights
        compromised_terminals = []
        for terminal_type, weight in terminal_type_weights.items():
            if weight > 0:
                type_terminals = terminal_profiles_table[
                    terminal_profiles_table.terminal_type == terminal_type
                ].TERMINAL_ID.values
                
                # Reduce the number of compromised terminals
                n_compromised = max(1, int(len(type_terminals) * weight * 0.001))  # Reduced from 0.01
                compromised_terminals.extend(
                    np.random.choice(type_terminals, size=n_compromised, replace=False)
                )
        
        # Find transactions from compromised terminals
        compromised_transactions = transactions_df[
            (transactions_df.TX_TIME_DAYS >= day) & 
            (transactions_df.TX_TIME_DAYS < day+45) & 
            (transactions_df.TERMINAL_ID.isin(compromised_terminals))
        ]
        
        # Apply fraud probability
        fraud_mask = compromised_transactions.index[
            np.random.random(len(compromised_transactions)) < FRAUD_CONFIG['fraud_probabilities']['compromised_terminal']
        ]
        
        transactions_df.loc[fraud_mask, 'TX_FRAUD'] = 1
        transactions_df.loc[fraud_mask, 'TX_FRAUD_SCENARIO'] = 2
    
    nb_frauds_scenario_2 = transactions_df.TX_FRAUD.sum() - nb_frauds_scenario_1
    print("Number of frauds from scenario 2 (Compromised Terminal): "+str(nb_frauds_scenario_2))
    
    # Scenario 3: Card-Not-Present Fraud (focusing on online terminals)
    for day in range(transactions_df.TX_TIME_DAYS.max()):
        # Target high-value customers more frequently
        high_value_customers = customer_profiles_table[
            customer_profiles_table.mean_amount > customer_profiles_table.mean_amount.quantile(0.7)
        ].CUSTOMER_ID.values
        
        # Select compromised customers, biased towards high-value ones
        n_high = np.random.randint(2, 4)  # 2-3 high-value customers
        n_regular = np.random.randint(1, 3)  # 1-2 regular customers
        
        compromised_customers = np.concatenate([
            np.random.choice(high_value_customers, size=min(n_high, len(high_value_customers)), replace=False),
            np.random.choice(customer_profiles_table.CUSTOMER_ID, size=n_regular, replace=False)
        ])
        
        # Focus on online terminals for CNP fraud
        online_terminals = terminal_profiles_table[terminal_profiles_table.terminal_type == 'online'].TERMINAL_ID.values
        
        compromised_transactions = transactions_df[
            (transactions_df.TX_TIME_DAYS >= day) & 
            (transactions_df.TX_TIME_DAYS < day+14) & 
            (transactions_df.CUSTOMER_ID.isin(compromised_customers)) &
            (transactions_df.TERMINAL_ID.isin(online_terminals))
        ]
        
        # Higher fraud rate during night hours (2 AM - 6 AM)
        night_hours_mask = ((compromised_transactions.TX_TIME_SECONDS % 86400) >= (2 * 3600)) & \
                          ((compromised_transactions.TX_TIME_SECONDS % 86400) <= (6 * 3600))
        
        if len(compromised_transactions[night_hours_mask]) > 0:
            transactions_df.loc[compromised_transactions[night_hours_mask].index, 'TX_AMOUNT'] *= \
                np.random.uniform(2, 4, size=len(compromised_transactions[night_hours_mask]))
            transactions_df.loc[compromised_transactions[night_hours_mask].index, 'TX_FRAUD'] = 1
            transactions_df.loc[compromised_transactions[night_hours_mask].index, 'TX_FRAUD_SCENARIO'] = 3
    
    nb_frauds_scenario_3 = transactions_df.TX_FRAUD.sum() - nb_frauds_scenario_2 - nb_frauds_scenario_1
    print("Number of frauds from scenario 3 (CNP Fraud): "+str(nb_frauds_scenario_3))
    
    # Scenario 4: Quick Cash-Out Fraud (rapid high-value transactions)
    for day in range(transactions_df.TX_TIME_DAYS.max()):
        # Target medium and high spender segments
        potential_targets = customer_profiles_table[
            customer_profiles_table.mean_amount > customer_profiles_table.mean_amount.median()
        ].CUSTOMER_ID.values
        
        compromised_customers = np.random.choice(
            potential_targets, 
            size=min(2, len(potential_targets)), 
            replace=False
        )
        
        for customer_id in compromised_customers:
            # Create burst of transactions targeting ATMs and retail terminals
            cash_out_terminals = terminal_profiles_table[
                terminal_profiles_table.terminal_type.isin(['atm', 'retail'])
            ].TERMINAL_ID.values
            
            if len(cash_out_terminals) > 0:
                # 3-5 rapid transactions
                n_transactions = np.random.randint(3, 6)
                base_time = np.random.randint(32400, 72000)  # Between 9 AM and 8 PM
                
                for i in range(n_transactions):
                    if len(transactions_df[transactions_df.CUSTOMER_ID == customer_id]) > 0:
                        template_tx = transactions_df[transactions_df.CUSTOMER_ID == customer_id].iloc[0].copy()
                        terminal_id = np.random.choice(cash_out_terminals)
                        
                        # Transactions 5-15 minutes apart
                        tx_time = base_time + (i * np.random.randint(300, 900))
                        tx_time = tx_time % 86400
                        
                        template_tx.TX_TIME_SECONDS = day * 86400 + tx_time
                        template_tx.TX_TIME_DAYS = day
                        template_tx.TERMINAL_ID = terminal_id
                        template_tx.TX_AMOUNT *= np.random.uniform(2, 3)  # More realistic amount increase
                        template_tx.TX_FRAUD = 1
                        template_tx.TX_FRAUD_SCENARIO = 4
                        
                        # Convert template_tx to DataFrame and concatenate
                        new_tx = pd.DataFrame([template_tx.to_dict()])
                        transactions_df = pd.concat([transactions_df, new_tx], ignore_index=True)
    
    nb_frauds_scenario_4 = transactions_df.TX_FRAUD.sum() - nb_frauds_scenario_3 - nb_frauds_scenario_2 - nb_frauds_scenario_1
    print("Number of frauds from scenario 4 (Quick Cash-Out): "+str(nb_frauds_scenario_4))
    
    # Sort by datetime after adding new transactions
    transactions_df = transactions_df.sort_values('TX_DATETIME')
    transactions_df.reset_index(drop=True, inplace=True)
    
    # Print final fraud statistics
    total_frauds = transactions_df.TX_FRAUD.sum()
    total_transactions = len(transactions_df)
    print(f"\nFinal fraud statistics:")
    print(f"Total transactions: {total_transactions}")
    print(f"Total frauds: {total_frauds}")
    print(f"Overall fraud rate: {total_frauds/total_transactions:.4%}")
    
    for scenario in range(1, 5):
        scenario_frauds = len(transactions_df[transactions_df.TX_FRAUD_SCENARIO == scenario])
        if total_frauds > 0:
            print(f"Scenario {scenario} proportion: {scenario_frauds/total_frauds:.2%}")
    
    return transactions_df

def generate_dataset(n_customers=5000, n_terminals=10000, nb_days=365, start_date="2024-01-01", r=7):
    start_time=time.time()
    customer_profiles_table = generate_customer_profiles_table(n_customers, random_state = 0)
    print("Time to generate customer profiles table: {0:.2}s".format(time.time()-start_time))
    
    start_time=time.time()
    terminal_profiles_table = generate_terminal_profiles_table(n_terminals, random_state = 1)
    print("Time to generate terminal profiles table: {0:.2}s".format(time.time()-start_time))
    
    start_time=time.time()
    x_y_terminals = terminal_profiles_table[['x_terminal_id','y_terminal_id']].values.astype(float)
    customer_profiles_table['available_terminals'] = customer_profiles_table.apply(lambda x : get_list_terminals_within_radius(x, x_y_terminals=x_y_terminals, r=r), axis=1)
    customer_profiles_table['nb_terminals']=customer_profiles_table.available_terminals.apply(len)
    print("Time to associate terminals to customers: {0:.2}s".format(time.time()-start_time))
    
    start_time=time.time()
    # Generate transactions for each customer
    all_transactions = []
    for _, customer in customer_profiles_table.iterrows():
        customer_txs = generate_transactions_table(customer, start_date=start_date, nb_days=nb_days)
        all_transactions.append(customer_txs)
    
    # Combine all transactions
    transactions_df = pd.concat(all_transactions, ignore_index=True)
    print("Time to generate transactions: {0:.2}s".format(time.time()-start_time))
    
    # Sort transactions chronologically
    transactions_df=transactions_df.sort_values('TX_DATETIME')
    # Reset indices, starting from 0
    transactions_df.reset_index(inplace=True,drop=True)
    transactions_df.reset_index(inplace=True)
    # TRANSACTION_ID are the dataframe indices, starting from 0
    transactions_df.rename(columns = {'index':'TRANSACTION_ID'}, inplace = True)
    
    return (customer_profiles_table, terminal_profiles_table, transactions_df)

def save_generated_data(transactions_df, terminal_profiles_table, customer_profiles_table):
    """Save generated data to CSV files with proper directory handling"""
    # Ensure data directory exists
    data_dir = "./data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")
    
    # Save files with error handling
    try:
        transactions_df.to_csv(f"{data_dir}/transactions_df.csv", index=False)
        terminal_profiles_table.to_csv(f"{data_dir}/terminal_profiles_table.csv", index=False)
        customer_profiles_table.to_csv(f"{data_dir}/customer_profiles_table.csv", index=False)
        print("Successfully saved all data files to the data directory")
    except Exception as e:
        print(f"Error saving data files: {str(e)}")

# Main block to execute when the script runs directly
if __name__ == "__main__":
    # Configuration for the dataset
    N_CUSTOMERS = 100  # Increased for more realistic dataset
    N_TERMINALS = 200
    NB_DAYS = 30  # One month of data
    START_DATE = "2024-01-01"
    RADIUS = 7  # Radius for connecting customers to terminals
    
    print("Generating fraud detection dataset...")
    print(f"Parameters: {N_CUSTOMERS} customers, {N_TERMINALS} terminals, {NB_DAYS} days of data")
    
    # Generate the dataset
    customer_profiles, terminal_profiles, transactions = generate_dataset(
        n_customers=N_CUSTOMERS,
        n_terminals=N_TERMINALS,
        nb_days=NB_DAYS,
        start_date=START_DATE,
        r=RADIUS
    )
    
    # Add seasonal patterns to transactions
    transactions = add_seasonal_patterns(transactions, START_DATE)
    
    # Add fraudulent transactions
    transactions = add_frauds(customer_profiles, terminal_profiles, transactions)
    
    # Save the generated data
    save_generated_data(transactions, terminal_profiles, customer_profiles)
    
    print("Dataset generation complete!")
