import streamlit as st

from app.ui.tabs import (
    run_tab_ai_summary,
    run_tab_configure,
    run_tab_export,
    run_tab_feature_comparison,
    run_tab_metrics,
    run_tab_model,
    run_tab_predict,
    run_tab_select_data,
    run_tab_view,
    run_tab_visualize,
)


def main() -> None:
    """Main function to run the Streamlit application."""
    st.set_page_config(page_title="BenchML", page_icon=":bar_chart:")
    st.header(body="BenchML", divider="gray", width="content")
    st.markdown(
        body="""
    Benchmark machine learning models by injecting synthetic data, evaluating performance,
    and generating AI summaries. Additional information can be found on the [GitHub repository](https://github.com/miktc/benchml).
    """
    )

    tab = st.tabs(
        tabs=[
            "Model",
            "Select Data",
            "Configure",
            "View",
            "Metrics",
            "Predict",
            "Visualize",
            "Feature Comparison",
            "Export",
            "AI Summary",
        ]
    )

    with tab[0]:
        run_tab_model()

    with tab[1]:
        run_tab_select_data()

    with tab[2]:
        run_tab_configure()

    with tab[3]:
        run_tab_view()

    with tab[4]:
        run_tab_metrics()

    with tab[5]:
        run_tab_predict()

    with tab[6]:
        run_tab_visualize()

    with tab[7]:
        run_tab_feature_comparison()

    with tab[8]:
        run_tab_export()

    with tab[9]:
        run_tab_ai_summary()

    st.divider()


if __name__ == "__main__":
    main()
