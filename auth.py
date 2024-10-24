import streamlit as st
import requests

# URL of the API (adjust this as needed)
API_URL = "http://localhost:8000"  # Adjust according to your actual API endpoint

# Session state initialization
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'token' not in st.session_state:
    st.session_state.token = None

def register():
    st.title("Register")
    
    username = st.text_input("Username")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    
    if st.button("Register"):
        if password != confirm_password:
            st.error("Passwords do not match!")
        else:
            payload = {"username": username, "email": email, "password": password}
            response = requests.post(f"{API_URL}/auth/register", json=payload)
            
            if response.status_code == 201:
                st.success("Registration successful. Please log in.")
                st.experimental_rerun()
            else:
                st.error("Registration failed. Please try again.")

def login():
    st.title("Login")
    
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        payload = {"username": username, "password": password}
        response = requests.post(f"{API_URL}/auth/login", json=payload)
        
        if response.status_code == 200:
            token = response.json().get("access_token")
            st.session_state.token = token
            st.session_state.logged_in = True
            st.success("Login successful.")
            st.experimental_rerun()
        else:
            st.error("Login failed. Please check your credentials and try again.")

def main():
    if st.session_state.logged_in:
        st.write("You are logged in.")
        st.write("Redirecting to main page...")
        st.stop()
        import leafguard.main as main  # Import the main module
        main.main()  # Call the main function in main.py
    else:
        st.write("You are not logged in.")
        st.write("Please log in or register.")

# Main Streamlit app logic
def app():
    st.sidebar.title("Auth System")
    choice = st.sidebar.selectbox("Choose an option", ["Login", "Register"])

    if choice == "Login":
        login()
    else:
        register()

    main()

if __name__ == "__main__":
    app()
