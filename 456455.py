import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf  
from tensorflow.keras import layers, models  # type: ignore
import random
from datetime import datetime, timedelta
import json
import os
import pickle 
import sys  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  


SAVE_DIR = os.path.join(os.getcwd(), 'models')
os.makedirs(SAVE_DIR, exist_ok=True)  
print(f"Model will be saved to: {SAVE_DIR}")


def create_realistic_fitness_dataset(n_users=1000, n_exercises=300):
    print("Starting create_realistic_fitness_dataset")
    np.random.seed(42)
    random.seed(42)

    
    exercise_types = ['strength', 'cardio', 'flexibility']

    
    exercise_data = []

    
    strength_exercises = [
        ('Bench Press', 'chest', 'barbell', ['pectoralis', 'triceps', 'deltoids'], 3),
        ('Squat', 'legs', 'barbell', ['quadriceps', 'hamstrings', 'glutes'], 4),
        ('Deadlift', 'back', 'barbell', ['lower_back', 'hamstrings', 'glutes'], 5),
        ('Shoulder Press', 'shoulders', 'dumbbell', ['deltoids', 'triceps'], 3),
        ('Bicep Curl', 'arms', 'dumbbell', ['biceps'], 2),
        ('Tricep Extension', 'arms', 'cable', ['triceps'], 2),
        ('Lat Pulldown', 'back', 'machine', ['latissimus_dorsi', 'biceps'], 3),
        ('Leg Press', 'legs', 'machine', ['quadriceps', 'hamstrings', 'glutes'], 3),
        ('Push-ups', 'chest', 'bodyweight', ['pectoralis', 'triceps', 'deltoids'], 2),
        ('Pull-ups', 'back', 'bodyweight', ['latissimus_dorsi', 'biceps'], 4),
        ('Dips', 'chest', 'bodyweight', ['pectoralis', 'triceps'], 3),
        ('Romanian Deadlift', 'back', 'barbell', ['hamstrings', 'glutes', 'lower_back'], 4),
        ('Overhead Press', 'shoulders', 'barbell', ['deltoids', 'triceps'], 3),
        ('Rows', 'back', 'dumbbell', ['latissimus_dorsi', 'rhomboids'], 3)
    ]

    
    cardio_exercises = [
        ('Running', 'full_body', 'none', ['legs', 'heart'], 3),
        ('Cycling', 'legs', 'machine', ['quadriceps', 'hamstrings', 'heart'], 2),
        ('Swimming', 'full_body', 'none', ['arms', 'back', 'legs', 'heart'], 3),
        ('Jump Rope', 'full_body', 'equipment', ['legs', 'shoulders', 'heart'], 3),
        ('Burpees', 'full_body', 'none', ['full_body', 'heart'], 4),
        ('Mountain Climbers', 'full_body', 'none', ['core', 'shoulders', 'heart'], 3),
        ('High Knees', 'legs', 'none', ['legs', 'core', 'heart'], 2),
        ('Jumping Jacks', 'full_body', 'none', ['full_body', 'heart'], 2)
    ]

    
    flexibility_exercises = [
        ('Hamstring Stretch', 'legs', 'none', ['hamstrings'], 1),
        ('Quad Stretch', 'legs', 'none', ['quadriceps'], 1),
        ('Hip Flexor Stretch', 'legs', 'none', ['hip_flexors'], 1),
        ('Shoulder Stretch', 'shoulders', 'none', ['deltoids'], 1),
        ('Cat-Cow Stretch', 'back', 'none', ['spine', 'core'], 1),
        ('Cobra Stretch', 'back', 'none', ['lower_back'], 1),
        ("Child's Pose", 'back', 'none', ['back', 'shoulders'], 1),
        ('Downward Dog', 'full_body', 'none', ['shoulders', 'hamstrings', 'calves'], 2)
    ]

    
    all_exercises = strength_exercises + cardio_exercises + flexibility_exercises

    
    equipment_variations = ['barbell', 'dumbbell', 'kettlebell', 'machine', 'cable', 'resistance_band', 'bodyweight', 'none']
    intensity_variations = ['beginner', 'intermediate', 'advanced']

    
    generated_names = set(name for name, _, _, _, _ in all_exercises)

    
    print("Generating base exercises")
    for i, (name, primary_muscle, equipment, muscles_worked, difficulty) in enumerate(all_exercises):
        if i < len(strength_exercises):
            ex_type = 'strength'
        elif i < len(strength_exercises) + len(cardio_exercises):
            ex_type = 'cardio'
        else:  # flexibility
            ex_type = 'flexibility'

        exercise_id = i + 1

        
        if ex_type == 'cardio':
            met_value = np.random.uniform(5.0, 10.0)
        elif ex_type == 'strength':
            met_value = np.random.uniform(3.0, 7.0)
        else:  # flexibility
            met_value = np.random.uniform(2.0, 4.0)

        weight_bearing = equipment not in ['none', 'bodyweight']
        calories_per_minute = met_value * 3.5 * 70 / 200

        exercise_data.append({
            'exercise_id': exercise_id,
            'name': name,
            'type': ex_type,
            'primary_muscle': primary_muscle,
            'equipment': equipment,
            'muscles_worked': muscles_worked,
            'difficulty': intensity_variations[min(difficulty - 1, 2)],
            'met_value': round(met_value, 2),
            'weight_bearing': weight_bearing,
            'calories_per_minute': round(calories_per_minute, 2)
        })
    print(f"Generated {len(exercise_data)} base exercises")

    
    print("Generating exercise variations")
    exercise_id = len(all_exercises) + 1
    variation_count = 0
    max_iterations = n_exercises * 10  # Prevent infinite loops
    iterations = 0
    
    while len(exercise_data) < n_exercises and iterations < max_iterations:
        iterations += 1
        base_exercise = random.choice(exercise_data[:len(all_exercises)])
        equipment_var = random.choice(equipment_variations)
        intensity_var = random.choice(intensity_variations)

        if base_exercise['type'] == 'flexibility' and equipment_var not in ['none', 'resistance_band', 'mat']:
            continue

        new_name = f"{equipment_var.title()} {base_exercise['name']}" if equipment_var != 'none' else base_exercise['name']
        new_name = f"{intensity_var.title()} {new_name}"

        if new_name in generated_names:
            continue

        generated_names.add(new_name)
        variation_count += 1

        variation = base_exercise.copy()
        variation['exercise_id'] = exercise_id
        variation['name'] = new_name
        variation['equipment'] = equipment_var
        variation['difficulty'] = intensity_var

        intensity_factor = 0.8 if intensity_var == 'beginner' else 1.0 if intensity_var == 'intermediate' else 1.2
        variation['met_value'] = round(variation['met_value'] * intensity_factor, 2)
        variation['calories_per_minute'] = round(variation['calories_per_minute'] * intensity_factor, 2)

        exercise_data.append(variation)
        exercise_id += 1
        
        if variation_count % 50 == 0:
            print(f"  Generated {variation_count} variations so far")
    
    print(f"Generated {variation_count} exercise variations")
    print(f"Total exercises: {len(exercise_data)}")

    
    if len(exercise_data) < n_exercises:
        print(f"Warning: Could only generate {len(exercise_data)} exercises instead of {n_exercises}")

    exercises_df = pd.DataFrame(exercise_data[:min(len(exercise_data), n_exercises)])

    
    print("Generating user profiles")
    user_data = []
    fitness_levels = ['beginner', 'intermediate', 'advanced']
    goals = ['weight_loss', 'muscle_gain', 'endurance', 'flexibility', 'general_fitness']

    for user_id in range(1, n_users + 1):
        user_data.append({
            'user_id': user_id,
            'fitness_level': random.choice(fitness_levels),
            'goal': random.choice(goals),
            'age': random.randint(18, 65),
            'weight': random.uniform(50, 100),  # in kg
            'height': random.uniform(150, 190)  # in cm
        })

    users_df = pd.DataFrame(user_data)
    print(f"Generated {len(users_df)} user profiles")

    
    print("Generating workout history")
    workout_history = []
    start_date = datetime.now() - timedelta(days=180)  # Last 6 months

    for user_id in range(1, n_users + 1):
        if user_id % 100 == 0:
            print(f"  Generating workout history for user {user_id}/{n_users}")
        n_workouts = random.randint(10, 100)  # Number of workouts per user
        for _ in range(n_workouts):
            workout_date = start_date + timedelta(
                days=random.randint(0, 180),
                hours=random.randint(6, 20)
            )

            
            for _ in range(random.randint(3, 8)):
                exercise_id = random.randint(1, len(exercises_df))
                duration = random.randint(5, 30)  # minutes

                workout_history.append({
                    'user_id': user_id,
                    'exercise_id': exercise_id,
                    'date': workout_date,
                    'duration': duration,
                    'completed': random.random() > 0.1  # 90% completion rate
                })

    workout_history_df = pd.DataFrame(workout_history)
    print(f"Generated {len(workout_history_df)} workout history records")

    
    print("Generating ratings")
    ratings = []
    for user_id in range(1, n_users + 1):
        if user_id % 100 == 0:
            print(f"  Generating ratings for user {user_id}/{n_users}")
        
        n_ratings = random.randint(10, 30)
        rated_exercises = random.sample(range(1, len(exercises_df) + 1), min(n_ratings, len(exercises_df)))

        for exercise_id in rated_exercises:
            ratings.append({
                'user_id': user_id,
                'exercise_id': exercise_id,
                'rating': random.randint(1, 5),
                'date': start_date + timedelta(days=random.randint(0, 180))
            })

    ratings_df = pd.DataFrame(ratings)
    print(f"Generated {len(ratings_df)} ratings")

    print("Finished creating dataset")
    
    return {
        'users': users_df,
        'exercises': exercises_df,
        'workout_history': workout_history_df,
        'ratings': ratings_df
    }



class WorkoutRecommender:
    def __init__(self, dataset):
        print("Initializing WorkoutRecommender")
        self.users_df = dataset['users']
        self.exercises_df = dataset['exercises']
        self.workout_history_df = dataset['workout_history']
        self.ratings_df = dataset['ratings']

        
        self.fitness_level_encoder = LabelEncoder()
        self.goal_encoder = LabelEncoder()
        self.exercise_type_encoder = LabelEncoder()
        self.difficulty_encoder = LabelEncoder()

        
        print("Fitting encoders")
        self.fitness_level_encoder.fit(self.users_df['fitness_level'])
        self.goal_encoder.fit(self.users_df['goal'])
        self.exercise_type_encoder.fit(self.exercises_df['type'])
        self.difficulty_encoder.fit(self.exercises_df['difficulty'])
        print("Finished fitting encoders")

        
        print("Checking for existing model")
        model_loaded = self.load_model()

        if not model_loaded:
            print("No existing model found. Training a new model...")
            # Train the model
            self._prepare_training_data()
            self._build_and_train_model()
            self.save_model()
        else:
            print("Using existing trained model")

    def _prepare_training_data(self):
        print("Preparing training data")
        try:
            training_data = pd.merge(
                self.workout_history_df,
                self.ratings_df[['user_id', 'exercise_id', 'rating']],
                on=['user_id', 'exercise_id'],
                how='left'
            )

            training_data['rating'] = training_data['rating'].fillna(3.0)

            training_data = pd.merge(
                training_data,
                self.users_df[['user_id', 'fitness_level', 'goal']],
                on='user_id'
            )
            training_data = pd.merge(
                training_data,
                self.exercises_df[['exercise_id', 'type', 'difficulty']],
                on='exercise_id'
            )

            training_data['fitness_level_encoded'] = self.fitness_level_encoder.transform(training_data['fitness_level'])
            training_data['goal_encoded'] = self.goal_encoder.transform(training_data['goal'])
            training_data['type_encoded'] = self.exercise_type_encoder.transform(training_data['type'])
            training_data['difficulty_encoded'] = self.difficulty_encoder.transform(training_data['difficulty'])

            self.X = training_data[[
                'fitness_level_encoded',
                'goal_encoded',
                'type_encoded',
                'difficulty_encoded'
            ]].values

            self.y = training_data['rating'].values

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42
            )

            self.scaler = StandardScaler()
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
            
            print(f"Prepared training data with {len(self.X_train)} training samples")
        except Exception as e:
            print(f"Error in _prepare_training_data: {str(e)}")
            raise

    def _build_and_train_model(self):
        print("Building and training model")
        try:
            tf.random.set_seed(42)
            
            self.model = models.Sequential([
                layers.Dense(64, activation='relu', input_shape=(4,)),
                layers.Dropout(0.2),
                layers.Dense(32, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(1) 
            ])
            self.model.compile(optimizer='adam', loss='mse')  # Use MSE for regression
            self.model.summary()

            self.model.fit(self.X_train, self.y_train, epochs=10, validation_data=(self.X_test, self.y_test), verbose=0)
            print("Finished training the model")
        except Exception as e:
            print(f"Error in _build_and_train_model: {str(e)}")
            raise
    
    def save_model(self):
        """Saves the trained model and scaler."""
        try:
            self.model.save(os.path.join(SAVE_DIR, 'workout_recommender_model.h5'))
            with open(os.path.join(SAVE_DIR, 'scaler.pkl'), 'wb') as f:
                pickle.dump(self.scaler, f)
            print("Model and scaler saved successfully!")
        except Exception as e:
            print(f"Error saving model: {e}")
            raise

    def load_model(self):
        """Loads the trained model and scaler."""
        try:
            model_path = os.path.join(SAVE_DIR, 'workout_recommender_model.h5')
            scaler_path = os.path.join(SAVE_DIR, 'scaler.pkl')
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                print("Loading existing model and scaler...")
                self.model = tf.keras.models.load_model(model_path)
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                return True
            else:
                print("No existing model found.")
                return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def predict_rating(self, user_info, exercise):
        """Predicts the rating a user would give to an exercise."""
        try:
            user_fitness_level_encoded = self.fitness_level_encoder.transform([user_info['fitness_level']])[0]
            user_goal_encoded = self.goal_encoder.transform([user_info['goal']])[0]
            exercise_type_encoded = self.exercise_type_encoder.transform([exercise['type']])[0]
            exercise_difficulty_encoded = self.difficulty_encoder.transform([exercise['difficulty']])[0]

            input_data = np.array([
                user_fitness_level_encoded,
                user_goal_encoded,
                exercise_type_encoded,
                exercise_difficulty_encoded
            ]).reshape(1, -1)

            input_data_scaled = self.scaler.transform(input_data)  # Use the loaded scaler
            predicted_rating = self.model.predict(input_data_scaled)[0][0]
            return predicted_rating
        except Exception as e:
            print(f"Error in predict_rating: {str(e)}")
            return 3.0 

    def get_top_exercises(self, user_info, exercises, n=5):
        """Gets the top-rated exercises for a user."""
        try:
            exercise_ratings = []
            for exercise in exercises:
                rating = self.predict_rating(user_info, exercise)
                exercise_ratings.append((exercise, rating))

            
            exercise_ratings.sort(key=lambda x: x[1], reverse=True)
            return [exercise for exercise, _ in exercise_ratings[:n]]
        except Exception as e:
            print(f"Error in get_top_exercises: {str(e)}")
            return []

    def create_weekly_plan(self, user_info, recommender):
        """Creates a weekly workout plan for a user."""
        try:
            available_exercises = self.exercises_df.to_dict(orient='records')
            
            
            if user_info['fitness_level'] == 'beginner':
                available_exercises = [ex for ex in available_exercises if ex['difficulty'] in ['beginner', 'intermediate']]
            elif user_info['fitness_level'] == 'intermediate':
                available_exercises = [ex for ex in available_exercises if ex['difficulty'] in ['intermediate', 'advanced']]
            top_exercises = recommender.get_top_exercises(user_info, available_exercises, n=50)  # Get more than needed

            # Further filter exercises based on the user's goal and available time
            if user_info['goal'] == 'weight_loss':
                eligible_exercises = [ex for ex in top_exercises if ex['type'] in ['cardio', 'strength']]
                eligible_exercises = sorted(eligible_exercises, key=lambda x: x['calories_per_minute'], reverse=True)
            elif user_info['goal'] == 'muscle_gain':
                eligible_exercises = [ex for ex in top_exercises if ex['type'] == 'strength']
            elif user_info['goal'] == 'endurance':
                eligible_exercises = [ex for ex in top_exercises if ex['type'] == 'cardio']
            elif user_info['goal'] == 'flexibility':
                eligible_exercises = [ex for ex in top_exercises if ex['type'] == 'flexibility']
            else: 
                eligible_exercises = top_exercises  

            if not eligible_exercises:
                eligible_exercises = top_exercises
            
            
            num_days = user_info['workout_days']
            exercises_per_day = 5
            selected_exercises = []
            
            
            num_exercises_to_select = min(len(eligible_exercises), num_days * exercises_per_day)
            
            
            selected_exercises = eligible_exercises[:num_exercises_to_select]
            
            
            weekly_plan = {}
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            for i in range(num_days):
                day_name = days[i]
                day_exercises = selected_exercises[i * exercises_per_day : (i + 1) * exercises_per_day]
                
                if not day_exercises:
                    day_exercises = selected_exercises[i % len(selected_exercises)]
                
                
                total_calories = sum(ex['calories_per_minute'] * user_info['workout_duration'] for ex in day_exercises)
                
                target_muscles = set()
                for ex in day_exercises:
                    target_muscles.update(ex['muscles_worked'])
                
                
                weekly_plan[day_name] = {
                    'focus': user_info['goal'],
                    'target_muscles': list(target_muscles),
                    'workout': [{
                        'name': ex['name'],
                        'sets': random.randint(2, 4),
                        'reps': random.randint(8, 15)
                    } for ex in day_exercises],
                    'total_calories': round(total_calories, 2)
                }
            
            return weekly_plan
        except Exception as e:
            print(f"Error in create_weekly_plan: {str(e)}")
            return {}


def serialize_np_and_df(obj):
    """JSON serializer for numpy arrays and pandas DataFrames."""
    if isinstance(obj, (np.ndarray, pd.DataFrame)):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")



def get_user_input():
    """Collect user preferences for workout plan generation."""

    print("Welcome to the Workout Planner!")
    fitness_level = input("Enter your fitness level (beginner, intermediate, advanced): ").lower()
    while fitness_level not in ['beginner', 'intermediate', 'advanced']:
        print("Invalid input. Please enter beginner, intermediate, or advanced.")
        fitness_level = input("Enter your fitness level: ").lower()

    goal = input("Enter your fitness goal (weight_loss,muscle_gain, endurance, flexibility, general_fitness): ").lower()
    while goal not in ['weight_loss', 'muscle_gain', 'endurance', 'flexibility', 'general_fitness']:
        print("Invalid input. Please enter a valid goal.")
        goal = input("Enter your fitness goal: ").lower()

    workout_days = int(input("Enter how many days a week you want to workout (3-7): "))
    while not 3 <= workout_days <= 7:
        print("Invalid input. Please enter a number between 3 and 7.")
        workout_days = int(input("Enter how many days a week you want to workout (3-7): "))

    workout_duration = int(input("Enter your preferred workout duration in minutes: "))
    while not 15 <= workout_duration <= 120:  #reasonable limits
        print("Invalid input. Please enter a duration between 15 and 120 minutes.")
        workout_duration = int(input("Enter your preferred workout duration in minutes: "))

    user_info = {
        'fitness_level': fitness_level,
        'goal': goal,
        'workout_days': workout_days,
        'workout_duration': workout_duration
    }
    return user_info



def main():
    """Main function to run the program"""
    try:
        print("Starting fitness workout planner program")

        
        dataset_files = ['users.pkl', 'exercises.pkl', 'workout_history.pkl', 'ratings.pkl']
        datasets_exist = all(os.path.exists(os.path.join(SAVE_DIR, f)) for f in dataset_files)

        if datasets_exist:
            print("Loading existing datasets")
            dataset = {}
            for file in dataset_files:
                with open(os.path.join(SAVE_DIR, file), 'rb') as f:
                    dataset[file.split('.')[0]] = pickle.load(f)
        else:
            print("Generating new datasets")
            dataset = create_realistic_fitness_dataset()

            
            os.makedirs(SAVE_DIR, exist_ok=True)
            for name, df in dataset.items():
                with open(os.path.join(SAVE_DIR, f"{name}.pkl"), 'wb') as f:
                    pickle.dump(df, f)

        
        print("Initializing recommender")
        recommender = WorkoutRecommender(dataset)

        
        user_info = get_user_input()

        
        print("Generating weekly plan for the user")
        weekly_plan = recommender.create_weekly_plan(user_info, recommender)

        
        print("Saving weekly plan to JSON")
        with open(os.path.join(SAVE_DIR, 'user_weekly_plan.json'), 'w') as f:
            json.dump(weekly_plan, f, default=serialize_np_and_df, indent=2)

        print("Program completed successfully!")
        print(f"User's weekly plan saved to {os.path.join(SAVE_DIR, 'user_weekly_plan.json')}")

        
        print("\nYour Complete Weekly Workout Plan:")
        print("==================================")
        for day, plan in weekly_plan.items():
            print(f"\n{day}:")
            print(f"Focus: {plan['focus']}")
            print(f"Target muscles: {', '.join(plan['target_muscles'])}")
            print("Exercises:")
            for i, exercise in enumerate(plan['workout'], 1):
                print(f"  {i}. {exercise['name']} - {exercise['sets']} sets of {exercise['reps']}")
            print(f"Total calories: {plan['total_calories']}")
            print("-" * 50)

    except Exception as e:
        print(f"Error in main function: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)



if __name__ == "__main__":
    main()