mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"ilyastarhouchi@gmail.com\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
