import streamlit as st
from IPC_backend import suggest_sections

st.title("🔍 IPC Section Suggestion System")
st.write("Enter the crime description below and get the most relevant IPC sections.")

# Input box
complaint = st.text_area("📝 Crime Description", height=200)

# Button
if st.button("Get IPC Suggestions"):
    if complaint.strip():
        with st.spinner("Analyzing crime description..."):
            suggestions = suggest_sections(complaint)

        if suggestions:
            st.success("✅ Suggested IPC Sections:")

            for i, suggestion in enumerate(suggestions, start=1):
                st.markdown(f"""
                ### 🔹 Suggestion {i}
                **Description:** {suggestion['Description']}  
                **Offense:** {suggestion['Offense']}  
                **Punishment:** {suggestion['Punishment']}  
                **Cognizable:** {suggestion['Cognizable']}  
                **Bailable:** {suggestion['Bailable']}  
                **Court:** {suggestion['Court']}  
                ---
                """)
        else:
            st.error("❌ No matching records found.")
    else:
        st.warning("⚠ Please enter a valid crime description.")
