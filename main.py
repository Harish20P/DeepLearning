import streamlit as st
import numpy as np
import pickle

model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
label_encoder = pickle.load(open('label.pkl', 'rb'))

def generate_random_features():
    """Generate random features for the model within specified min and max values."""
    area = np.random.uniform(20420.0, 254616.0)
    perimeter = np.random.uniform(524.736, 1985.37)
    major_axis_length = np.random.uniform(183.601165, 738.8601535)
    minor_axis_length = np.random.uniform(122.5126535, 460.1984968)
    aspect_ratio = np.random.uniform(1.024867596, 2.430306447)
    eccentricity = np.random.uniform(0.218951263, 0.911422968)
    convex_area = np.random.uniform(20684.0, 263261.0)
    equiv_diameter = np.random.uniform(161.2437642, 569.3743583)
    extent = np.random.uniform(0.555314717, 0.866194641)
    solidity = np.random.uniform(0.919246157, 0.9946775)
    roundness = np.random.uniform(0.489618256, 0.9906854)
    compactness = np.random.uniform(0.640576759, 0.987302969)
    shape_factor1 = np.random.uniform(0.002778013, 0.010451169)
    shape_factor2 = np.random.uniform(0.000564169, 0.003664972)
    shape_factor3 = np.random.uniform(0.410338584, 0.974767153)
    shape_factor4 = np.random.uniform(0.947687403, 0.99973253)
    
    return np.array([area, perimeter, major_axis_length, minor_axis_length,
                     aspect_ratio, eccentricity, convex_area, equiv_diameter,
                     extent, solidity, roundness, compactness,
                     shape_factor1, shape_factor2, shape_factor3, shape_factor4])

def main():
    st.title("Dry Bean Class Prediction Web App")

    st.write("### Input Features")
    
    # Check if session state for features exists, if not generate random features
    if 'features' not in st.session_state:
        st.session_state.features = generate_random_features()

    # Display input fields with features
    features = st.session_state.features

    area = st.number_input("Area", value=float(features[0]), min_value=20420.0, max_value=254616.0, step=1.0)
    perimeter = st.number_input("Perimeter", value=float(features[1]), min_value=524.736, max_value=1985.37, step=0.01)
    major_axis_length = st.number_input("Major Axis Length", value=float(features[2]), min_value=183.601165, max_value=738.8601535, step=0.01)
    minor_axis_length = st.number_input("Minor Axis Length", value=float(features[3]), min_value=122.5126535, max_value=460.1984968, step=0.01)
    aspect_ratio = st.number_input("Aspect Ratio", value=float(features[4]), min_value=1.024867596, max_value=2.430306447, step=0.01)
    eccentricity = st.number_input("Eccentricity", value=float(features[5]), min_value=0.218951263, max_value=0.911422968, step=0.01)
    convex_area = st.number_input("Convex Area", value=float(features[6]), min_value=20684.0, max_value=263261.0, step=0.01)
    equiv_diameter = st.number_input("Equivalent Diameter", value=float(features[7]), min_value=161.2437642, max_value=569.3743583, step=0.01)
    extent = st.number_input("Extent", value=float(features[8]), min_value=0.555314717, max_value=0.866194641, step=0.01)
    solidity = st.number_input("Solidity", value=float(features[9]), min_value=0.919246157, max_value=0.9946775, step=0.01)
    roundness = st.number_input("Roundness", value=float(features[10]), min_value=0.489618256, max_value=0.9906854, step=0.01)
    compactness = st.number_input("Compactness", value=float(features[11]), min_value=0.640576759, max_value=0.987302969, step=0.01)
    shape_factor1 = st.number_input("Shape Factor 1", value=float(features[12]), min_value=0.002778013, max_value=0.010451169, step=0.0001)
    shape_factor2 = st.number_input("Shape Factor 2", value=float(features[13]), min_value=0.000564169, max_value=0.003664972, step=0.0001)
    shape_factor3 = st.number_input("Shape Factor 3", value=float(features[14]), min_value=0.410338584, max_value=0.974767153, step=0.01)
    shape_factor4 = st.number_input("Shape Factor 4", value=float(features[15]), min_value=0.947687403, max_value=0.99973253, step=0.01)

    if st.button("Predict Class"):
        input_features = [
            area, perimeter, major_axis_length, minor_axis_length,
            aspect_ratio, eccentricity, convex_area, equiv_diameter,
            extent, solidity, roundness, compactness,
            shape_factor1, shape_factor2, shape_factor3, shape_factor4
        ]

        # Scale the input features
        input_features_scaled = scaler.transform([input_features])

        # Make prediction
        prediction = model.predict(input_features_scaled)
        predicted_class_index = np.argmax(prediction, axis=1)

        # Get the predicted label
        predicted_label = label_encoder.inverse_transform(predicted_class_index)

        # Display the predicted class
        st.write(f"### Predicted Class: {predicted_label[0]}")

    # Button to regenerate random features
    if st.button("Generate Random Input"):
        st.session_state.features = generate_random_features()  
        st.rerun()  # Use st.rerun() to refresh the app

if __name__ == '__main__':
    main()
