import streamlit as st

def inject_global_css():
    st.markdown("""
    <style>
        /* ── Global Settings & Typography ── */
        .main .block-container {
            padding-top: 1rem;
            max-width: 1200px;
        }

        /* Hero title with gradient animation */
        @keyframes gradientShift {
            0%   { background-position: 0% 50%; }
            50%  { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        .hero-title {
            font-size: 2.6rem;
            font-weight: 800;
            background: linear-gradient(135deg, #06b6d4, #3b82f6, #8b5cf6, #06b6d4);
            background-size: 300% 300%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: gradientShift 4s ease infinite;
            margin-bottom: 0.2rem;
            letter-spacing: -0.5px;
        }
        
        .hero-title-predict { 
            background-image: linear-gradient(135deg, #667eea, #764ba2, #f093fb, #667eea); 
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .hero-title-about { 
            background-image: linear-gradient(135deg, #10b981, #3b82f6, #8b5cf6, #10b981); 
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .hero-subtitle {
            font-size: 1.05rem;
            color: #9ca3af;
            margin-bottom: 1.5rem;
            line-height: 1.6;
        }

        /* Section divider */
        .section-divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.3), transparent);
            margin: 1.5rem 0;
            border: none;
        }

        /* Glass card styling */
        .glass-card {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 20px;
            padding: 1.8rem;
            margin: 0.8rem 0;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .glass-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 40px rgba(59, 130, 246, 0.15);
        }
        
        /* Remove ugly gaps above headers inside glass cards */
        .glass-card h1, .glass-card h2, .glass-card h3, .glass-card h4, .glass-card h5 {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }

        /* Upload zone */
        [data-testid="stFileUploader"] {
            border: 2px dashed rgba(59, 130, 246, 0.4);
            border-radius: 16px;
            padding: 1rem;
            background: rgba(59, 130, 246, 0.05);
            transition: all 0.3s ease;
        }
        [data-testid="stFileUploader"]:hover {
            border-color: rgba(59, 130, 246, 0.8);
            background: rgba(59, 130, 246, 0.1);
        }

        /* Buttons & Badges */
        .prediction-badge {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            padding: 14px 28px;
            border-radius: 50px;
            font-size: 1.4rem;
            font-weight: 700;
            letter-spacing: 0.5px;
            margin: 0.5rem 0;
            animation: slideIn 0.5s ease;
        }
        @keyframes slideIn {
            from { opacity: 0; transform: translateX(-20px); }
            to   { opacity: 1; transform: translateX(0); }
        }
        .badge-dog {
            background: linear-gradient(135deg, #f59e0b, #d97706);
            color: #fff;
            box-shadow: 0 6px 24px rgba(245, 158, 11, 0.35);
        }
        .badge-cat {
            background: linear-gradient(135deg, #8b5cf6, #6d28d9);
            color: #fff;
            box-shadow: 0 6px 24px rgba(139, 92, 246, 0.35);
        }

        /* Confidence meter */
        .confidence-container { margin: 1rem 0; }
        .confidence-label { font-size: 0.9rem; color: #9ca3af; margin-bottom: 6px; }
        .confidence-bar-bg {
            width: 100%; height: 12px;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 10px; overflow: hidden;
        }
        .confidence-bar-fill {
            height: 100%; border-radius: 10px;
            transition: width 1s ease;
        }
        .confidence-value { font-size: 2rem; font-weight: 800; margin-top: 4px; }

        /* Animated upload hint */
        @keyframes float {
            0%, 100% { transform: translateY(0); }
            50%      { transform: translateY(-10px); }
        }
        .upload-hint {
            text-align: center;
            padding: 3rem 2rem;
            border-radius: 20px;
            background: rgba(59, 130, 246, 0.04);
            border: 1px dashed rgba(59, 130, 246, 0.2);
        }
        .upload-icon {
            font-size: 4rem;
            animation: float 3s ease-in-out infinite;
            display: inline-block;
        }
        .upload-hint-text {
            color: #9ca3af;
            font-size: 1rem;
            margin-top: 1rem;
        }

        /* Info banner */
        .info-banner {
            background: linear-gradient(135deg, rgba(6, 182, 212, 0.08), rgba(59, 130, 246, 0.08));
            border-left: 4px solid #3b82f6;
            border-radius: 0 12px 12px 0;
            padding: 1rem 1.2rem;
            margin: 1rem 0;
            font-size: 0.9rem;
            color: #d1d5db;
            line-height: 1.6;
        }
        
        /* Metric overriding for performance page */
        [data-testid="stMetricValue"] {
            font-size: 2rem;
            background: linear-gradient(135deg, #06b6d4, #3b82f6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        /* Download button styling */
        [data-testid="stDownloadButton"] > button {
            background: linear-gradient(135deg, #667eea, #764ba2) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.6rem 1.5rem !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
        }
        [data-testid="stDownloadButton"] > button:hover {
            transform: scale(1.03) !important;
            box-shadow: 0 6px 24px rgba(102, 126, 234, 0.4) !important;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: rgba(17, 24, 39, 0.95);
        }
    </style>
    """, unsafe_allow_html=True)

def render_sidebar_branding():
    """Adds branding and author info to the sidebar on every page."""
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align: center; margin-bottom: 25px;">
                <h2 style="background: linear-gradient(135deg, #06b6d4, #3b82f6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 900; margin-bottom: 0; font-size: 1.8rem; letter-spacing: -0.5px;">Dog vs Cat</h2>
                <div style="color: #9ca3af; margin-top: 2px; font-weight: 500; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 1.5px;">CNN Classifier</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown(
            """
            <div style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 16px; padding: 18px; margin-bottom: 25px; box-shadow: 0 4px 20px rgba(0,0,0,0.1);">
                <div style="font-size: 0.75rem; color: #6b7280; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 12px; font-weight: 700;">Model Stats</div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <span style="color: #9ca3af; font-size: 0.9rem;">Architecture</span>
                    <span style="color: #60a5fa; font-weight: 600; font-size: 0.9rem; background: rgba(59, 130, 246, 0.1); padding: 2px 8px; border-radius: 12px;">CNN V1</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <span style="color: #9ca3af; font-size: 0.9rem;">Input Size</span>
                    <span style="color: #e5e7eb; font-weight: 500; font-size: 0.9rem;">256x256 RGB</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <span style="color: #9ca3af; font-size: 0.9rem;">Classes</span>
                    <span style="color: #e5e7eb; font-weight: 500; font-size: 0.9rem;">Dog, Cat</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: #9ca3af; font-size: 0.9rem;">Parameters</span>
                    <span style="color: #10b981; font-weight: 600; font-size: 0.9rem;">~1.5M</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown(
            """
            <div style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 16px; padding: 18px; text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.1);">
                <div style="font-size: 0.75rem; color: #6b7280; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 8px; font-weight: 700; text-align: left;">Developer</div>
                <div style="font-size: 1.3rem; font-weight: 800; color: #f3f4f6; margin-bottom: 15px; background: linear-gradient(135deg, #e5e7eb, #9ca3af); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Abhay Mishra</div>
                <div style="display: flex; flex-direction: column; gap: 10px;">
                    <a href="https://www.linkedin.com/in/abhay-mishra-640286365?utm_source=share_via&utm_content=profile&utm_medium=member_android" target="_blank" style="text-decoration: none;">
                        <div style="background: linear-gradient(135deg, #0077b5, #005f8d); color: white; padding: 10px; border-radius: 8px; font-size: 0.9rem; font-weight: 600; transition: transform 0.2s;">
                            LinkedIn Profile
                        </div>
                    </a>
                    <a href="https://github.com/buddhu22" target="_blank" style="text-decoration: none;">
                        <div style="background: linear-gradient(135deg, #333333, #1a1a2e); border: 1px solid rgba(255,255,255,0.1); color: white; padding: 10px; border-radius: 8px; font-size: 0.9rem; font-weight: 600; transition: transform 0.2s;">
                            GitHub Profile
                        </div>
                    </a>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
