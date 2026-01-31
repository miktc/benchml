import openai
import streamlit as st

from app.core.central import CENTRAL
from typing import cast


def run_tab_ai_summary() -> None:
    """Renders the UI for the AI Summary tab.

    Notes:
        If the required session state keys are None, a Streamlit error message is displayed using `st.error`
        and the function returns early.
    """
    state = cast(CENTRAL, st.session_state)

    st.subheader(body="AI Summary", divider="gray", width="content")
    configure = state.get("configure")
    select_model = state.get("select_model")
    metrics = state.get("all_metrics")

    if not configure:
        st.info(body="Configuration is required!")
    else:
        if select_model is None or metrics is None:
            st.error(body="Session state keys must not be None to generate a summary!")
            return

        prompt = f"""
        A {select_model} machine learning model produced these metrics: {metrics}.
        Create a concise conclusion based on the provided metrics to highlight the key findings.
        """

        if "api_key" in state:
            api_key = st.text_input(
                label="Enter OpenAI API Key:",
                type="password",
                value=state.get("api_key", ""),
            )
            col1, col2 = st.columns(spec=[1, 0.28])
            col1.link_button(
                label="Generate API Key",
                url="https://platform.openai.com/account/api-keys",
                width=145,
            )
            submit_api_key = col2.button(label="Summarize", width=145)

            if api_key and submit_api_key:
                st.divider()
                try:
                    client = openai.OpenAI(api_key=api_key)
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a helpful assistant.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                    )
                    st.subheader(body="Summary:")
                    st.markdown(response.choices[0].message.content)
                except Exception:
                    st.error(
                        body="An error occurred: Check the entered key and try again!"
                    )
            elif submit_api_key and not api_key:
                st.warning(body="Enter OpenAI API Key!")
        else:
            api_key = st.text_input(label="Enter OpenAI API Key:", type="password")
            col1, col2 = st.columns(spec=[1, 0.28])
            col1.link_button(
                label="Generate API Key",
                url="https://platform.openai.com/account/api-keys",
                width=145,
            )
            submit_api_key = col2.button(label="Summarize", width=145)

            if api_key and submit_api_key:
                try:
                    hold_info = st.empty()
                    hold_info.info(body="Generating summary...")
                    client = openai.OpenAI(api_key=api_key)
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a helpful assistant.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                    )
                    st.divider()
                    st.subheader(body="Summary:")
                    st.markdown(response.choices[0].message.content)
                    hold_info.empty()
                    state["api_key"] = api_key
                except Exception:
                    st.error(
                        body="An error occurred: Check the entered key and try again!"
                    )
            elif submit_api_key and not api_key:
                st.warning(body="Enter OpenAI API Key!")
