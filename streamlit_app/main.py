import streamlit as st
from utils import setup_page
import pages.overview as overview

# Setup
setup_page("CoreTax Sentiment Dashboard")

# Show Overview by default on Main page
overview.show()

