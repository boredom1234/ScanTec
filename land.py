import streamlit as st

# Set up the page configuration
st.set_page_config(
    page_title="ScanTec",
    page_icon="🩺",
    layout="centered",
)

# Page Title and Description
st.markdown("<h1 style='text-align: center; margin-bottom: 20px;'>🩺 ScanTec</h1>", unsafe_allow_html=True)
st.markdown("""
<h2 style="text-align: center; color: #4B9CD3; margin-bottom: 40px;">🌟 Empowering Healthcare with AI-driven Insights</h2>

<p style="text-align: center; font-size: 18px; margin-bottom: 20px;">
Welcome to <strong>ScanTec</strong>, a cutting-edge product designed to enhance medical imaging interpretation.
Our product offers two advanced modules:
</p>

<div style="text-align: center; font-size: 18px; margin-bottom: 40px;">
    <ul style="list-style: none; padding-left: 0;">
        <li>🦴 <strong>Fracture Detection</strong></li>
        <li>🧬 <strong>Pathology Interpretation</strong></li>
    </ul>
</div>

<p style="text-align: center; font-size: 18px; margin-bottom: 40px;">
Click on the buttons below to explore each module.
</p>
""", unsafe_allow_html=True)

# Create buttons to switch between modules
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔍 Fracture Detection")
    st.markdown("""
    💯 <strong>High Accuracy:</strong> Detects fractures with precision, reducing false positives and negatives.<br>
    ⚡ <strong>Speed:</strong> Processes images in seconds, ensuring quick and reliable diagnoses.<br>
    👌 <strong>Ease of Use:</strong> Simple interface designed for seamless integration into your existing workflow.
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
  # Add a bit of vertical space
    st.markdown("""
    <div style="text-align: center; margin-top: 20px;">
        <a href="https://www.google.com" target="_self">
            <button style="background-color: #4B9CD3; color: white; border: none; padding: 10px 20px; font-size: 16px; border-radius: 5px; cursor: pointer;">
                Go to Fracture Detection
            </button>
        </a>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.subheader("🧫 Pathology Interpretation")
    st.markdown("""
    <strong>Comprehensive Analysis:</strong> Supports multiple types of pathologies, offering a broad range of diagnostic insights.<br>
    📊 <strong>AI-Assisted Reporting:</strong> Generates detailed reports, saving time and enhancing accuracy.<br>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)  # Add a bit of vertical space
    st.markdown("""
    <div style="text-align: center; margin-top: 20px;">
        <a href="https://www.google.com" target="_self">
            <button style="background-color: #4B9CD3; color: white; border: none; padding: 10px 20px; font-size: 16px; border-radius: 5px; cursor: pointer;">
                Go to Pathology Interpretation
            </button>
        </a>
    </div>
    """, unsafe_allow_html=True)

# Add some space before the call to action
st.markdown("", unsafe_allow_html=True)

# Call to Action
st.markdown("""
<hr style="border-top: 2px solid #4B9CD3; margin-top: 40px;">

<h3 style="text-align: center; color: #4B9CD3; margin-top: 20px;">🚀 Ready to Revolutionize Your Diagnostic Workflow?</h3>

<p style="text-align: center; font-size: 18px; margin-top: 20px;">
Explore how ScanTec can integrate into your practice and elevate your diagnostic capabilities.
</p>

<p style="text-align: center; font-size: 18px; margin-top: 20px;">
</p>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<hr style="border-top: 2px solid #4B9CD3; margin-top: 40px;">

<p style="text-align: center; color: #4B9CD3; margin-top: 20px;">© 2024 ScanTec | Empowering Healthcare with AI</p>
""", unsafe_allow_html=True)
