import streamlit as st

from app.core.central import CENTRAL
from typing import cast


def run_tab_metrics() -> None:
    """Renders the UI for the Metrics tab.

    Notes:
        If the required session state keys are None, a Streamlit error message is displayed using `st.error`
        and the function returns early.
    """
    state = cast(CENTRAL, st.session_state)

    st.subheader(body="Metrics", divider="gray", width="content")
    configure = state.get("configure")
    model_type_select = state.get("model_type_select")
    metrics = state.get("all_metrics")

    if not configure:
        st.info(body="Configuration is required!")
    else:
        precision_list = []
        recall_list = []
        f1_list = []
        accuracy_list = []
        count = 0

        # Not configured when modifying the Model tab or Select Data tab
        if state.get("update_config") != state.get("original_config"):
            state["configure"] = False
            st.info(body="Configuration is required!")
        else:
            if model_type_select is None and metrics is None:
                st.error(
                    body="Session state keys must not be None to render the 'Metrics' tab!"
                )
                return

            model_type_select = cast(str, model_type_select)
            metrics = cast(dict, metrics)

            if model_type_select == "Classification":
                for k, v in metrics.items():
                    if not hasattr(v, "items"):
                        accuracy_list.append(f"{k}: {round(v,4)}")
                    try:
                        precision_list.append(
                            f"{k} precision: {round(v["precision"],4)}"
                        )
                        recall_list.append(f"{k} recall: {round(v["recall"],4)}")
                        f1_list.append(f"{k} f1-score: {round(v["f1-score"],4)}")
                        count += 1
                    except TypeError:
                        pass

                select_metric = st.selectbox(
                    label="Select Metric:",
                    options=[
                        "F1-score",
                        "Precision",
                        "Recall",
                        "Accuracy",
                        "All Metrics",
                    ],
                )
                class_count = ["Select All"]
                for i in range(count - 2):
                    class_count.append(str(i))

                if select_metric != "Accuracy":
                    select_class = st.selectbox(
                        label="Select Class:", options=class_count
                    )
                    if select_class != "Select All":
                        select_class_int = int(select_class)

                st.divider()
                if select_metric == "F1-score":
                    st.subheader(body="F1-score")
                    if select_class != "Select All":
                        st.markdown(body=f"**{f1_list[select_class_int]}**")
                    else:
                        for metric in f1_list:
                            st.markdown(body=f"**{metric}**")
                elif select_metric == "Precision":
                    st.subheader(body="Precision")
                    if select_class != "Select All":
                        st.markdown(body=f"**{precision_list[select_class_int]}**")
                    else:
                        for metric in precision_list:
                            st.markdown(body=f"**{metric}**")
                elif select_metric == "Recall":
                    st.subheader(body="Recall")
                    if select_class != "Select All":
                        st.markdown(body=f"**{recall_list[select_class_int]}**")
                    else:
                        for metric in recall_list:
                            st.markdown(body=f"**{metric}**")
                elif select_metric == "Accuracy":
                    st.subheader(body="Accuracy")
                    st.markdown(body=f"**{accuracy_list[0]}**")
                else:
                    st.subheader(body="All Metrics")
                    if select_class == "Select All":
                        for k, v in metrics.items():
                            if not hasattr(v, "items"):
                                st.markdown(body=f"**{k}: {round(v,4)}**")
                            try:
                                st.markdown(
                                    body=f"""
                                **{k} precision: {str(round(v["precision"],4))}**  
                                **{k} recall: {str(round(v["recall"],4))}**  
                                **{k} f1-score: {str(round(v["f1-score"],4))}**  
                                """
                                )
                            except TypeError:
                                pass

                    all_metrics_list = []
                    for k, v in metrics.items():
                        try:
                            text = f"""
                            **{k} precision: {str(round(v["precision"],4))}**  
                            **{k} recall: {str(round(v["recall"],4))}**  
                            **{k} f1-score: {str(round(v["f1-score"],4))}**  
                            """
                            all_metrics_list.append(text)
                        except TypeError:
                            pass

                    if select_class != "Select All":
                        st.write(all_metrics_list[select_class_int])
            else:
                for k, v in metrics.items():
                    for _ in k:
                        clean_k = k.replace("_", " ")
                    st.markdown(body=f"**{clean_k}: {round(v,4)}**")
