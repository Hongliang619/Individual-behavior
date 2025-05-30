import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime, timedelta
import numpy as np
import hashlib

# Page configuration
st.set_page_config(
    page_title="多用户行为追踪器",
    page_icon="📊",
    layout="wide"
)

# Data file paths
USERS_FILE = "users.json"
USER_DATA_DIR = "user_data"
COLORS_FILE_TEMPLATE = "user_data/{}_colors.json"

# Predefined color palette
DEFAULT_COLORS = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
    '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
    '#F8C471', '#82E0AA', '#F1948A', '#85929E', '#D2B4DE'
]

# Create user data directory
os.makedirs(USER_DATA_DIR, exist_ok=True)

def hash_password(password):
    """Hash password for security"""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    """Load user accounts"""
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_users(users):
    """Save user accounts"""
    with open(USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(users, f, ensure_ascii=False, indent=2)

def get_user_data_file(username):
    """Get user-specific data file path"""
    return os.path.join(USER_DATA_DIR, f"{username}_data.json")

def get_user_colors_file(username):
    """Get user-specific colors file path"""
    return os.path.join(USER_DATA_DIR, f"{username}_colors.json")

def load_user_data(username):
    """Load user-specific data"""
    data_file = get_user_data_file(username)
    if os.path.exists(data_file):
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_user_data(username, data):
    """Save user-specific data"""
    data_file = get_user_data_file(username)
    with open(data_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_user_colors(username):
    """Load user-specific event color mapping"""
    colors_file = get_user_colors_file(username)
    if os.path.exists(colors_file):
        try:
            with open(colors_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_user_colors(username, colors):
    """Save user-specific event color mapping"""
    colors_file = get_user_colors_file(username)
    with open(colors_file, 'w', encoding='utf-8') as f:
        json.dump(colors, f, ensure_ascii=False, indent=2)

def get_event_color(event_name, color_map, username):
    """Get event color, assign new color if it's a new event"""
    if event_name not in color_map:
        used_colors = set(color_map.values())
        available_colors = [c for c in DEFAULT_COLORS if c not in used_colors]
        if available_colors:
            color_map[event_name] = available_colors[0]
        else:
            # Generate random color if predefined colors are exhausted
            import random
            color_map[event_name] = f"#{random.randint(0,255):02x}{random.randint(0,255):02x}{random.randint(0,255):02x}"
        save_user_colors(username, color_map)
    return color_map[event_name]

def authenticate_user():
    """Handle user authentication"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.current_user = None
    
    if not st.session_state.authenticated:
        st.title("🔐 用户登录/注册")
        
        tab1, tab2 = st.tabs(["登录", "注册"])
        
        with tab1:
            st.subheader("用户登录")
            login_username = st.text_input("用户名", key="login_username")
            login_password = st.text_input("密码", type="password", key="login_password")
            
            if st.button("登录", type="primary"):
                users = load_users()
                if login_username in users:
                    if users[login_username]['password'] == hash_password(login_password):
                        st.session_state.authenticated = True
                        st.session_state.current_user = login_username
                        st.success(f"欢迎回来，{login_username}！")
                        st.rerun()
                    else:
                        st.error("密码错误")
                else:
                    st.error("用户不存在")
        
        with tab2:
            st.subheader("新用户注册")
            reg_username = st.text_input("用户名", key="reg_username", help="用户名将作为您的唯一标识")
            reg_password = st.text_input("密码", type="password", key="reg_password")
            reg_confirm_password = st.text_input("确认密码", type="password", key="reg_confirm_password")
            reg_display_name = st.text_input("显示姓名", key="reg_display_name", help="可选，用于显示")
            
            if st.button("注册", type="primary"):
                if not reg_username or not reg_password:
                    st.error("用户名和密码不能为空")
                elif reg_password != reg_confirm_password:
                    st.error("两次输入的密码不一致")
                elif len(reg_password) < 6:
                    st.error("密码长度至少6位")
                else:
                    users = load_users()
                    if reg_username in users:
                        st.error("用户名已存在")
                    else:
                        users[reg_username] = {
                            'password': hash_password(reg_password),
                            'display_name': reg_display_name or reg_username,
                            'created_at': datetime.now().isoformat(),
                            'last_login': datetime.now().isoformat()
                        }
                        save_users(users)
                        st.success("注册成功！请登录")
        
        return False
    
    return True

def time_to_minutes(time_str):
    """Convert time string to minutes"""
    hours, minutes = map(int, time_str.split(':'))
    return hours * 60 + minutes

def minutes_to_time(minutes):
    """Convert minutes to time string"""
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours:02d}:{mins:02d}"

def validate_time(time_str):
    """Validate time format"""
    try:
        if ':' not in time_str:
            return False
        parts = time_str.split(':')
        if len(parts) != 2:
            return False
        hours, minutes = map(int, parts)
        return 0 <= hours <= 23 and 0 <= minutes <= 59
    except:
        return False

def merge_same_events(df):
    """Merge consecutive time periods for the same event"""
    if df.empty:
        return df
    
    # Sort by start time
    df_sorted = df.sort_values('Start Time').copy()
    df_sorted['Start Minutes'] = df_sorted['Start Time'].apply(time_to_minutes)
    df_sorted['End Minutes'] = df_sorted['End Time'].apply(lambda x: time_to_minutes(x))
    
    # Handle cross-day events
    for i in range(len(df_sorted)):
        if df_sorted.iloc[i]['End Minutes'] <= df_sorted.iloc[i]['Start Minutes']:
            df_sorted.iloc[i, df_sorted.columns.get_loc('End Minutes')] += 1440
    
    merged_events = []
    
    # Group by event type
    for event_name in df_sorted['Event'].unique():
        event_df = df_sorted[df_sorted['Event'] == event_name].copy()
        event_df = event_df.sort_values('Start Minutes')
        
        # Merge overlapping or consecutive time periods
        current_start = event_df.iloc[0]['Start Minutes']
        current_end = event_df.iloc[0]['End Minutes']
        all_notes = [event_df.iloc[0]['Notes']]
        total_duration = event_df.iloc[0]['Duration (Hours)']
        
        for i in range(1, len(event_df)):
            next_start = event_df.iloc[i]['Start Minutes']
            next_end = event_df.iloc[i]['End Minutes']
            
            # If events are consecutive or overlapping (with 5-minute tolerance)
            if next_start <= current_end + 5:
                current_end = max(current_end, next_end)
                all_notes.append(event_df.iloc[i]['Notes'])
                total_duration += event_df.iloc[i]['Duration (Hours)']
            else:
                # Save current merged event
                merged_events.append({
                    'Start Time': minutes_to_time(current_start % 1440),
                    'End Time': minutes_to_time(current_end % 1440),
                    'Event': event_name,
                    'Duration (Hours)': round(total_duration, 2),
                    'Notes': ' | '.join(filter(None, all_notes))
                })
                
                # Start new event
                current_start = next_start
                current_end = next_end
                all_notes = [event_df.iloc[i]['Notes']]
                total_duration = event_df.iloc[i]['Duration (Hours)']
        
        # Don't forget the last event
        merged_events.append({
            'Start Time': minutes_to_time(current_start % 1440),
            'End Time': minutes_to_time(current_end % 1440),
            'Event': event_name,
            'Duration (Hours)': round(total_duration, 2),
            'Notes': ' | '.join(filter(None, all_notes))
        })
    
    return pd.DataFrame(merged_events)

def create_gantt_chart(df, color_map, username, merge_events=True):
    """Create Gantt chart with option to merge same events"""
    if merge_events:
        df = merge_same_events(df)
    
    fig = go.Figure()
    
    # Get unique events for consistent y-axis positioning
    unique_events = df['Event'].unique()
    event_positions = {event: i for i, event in enumerate(unique_events)}
    
    # Track legend entries to avoid duplicates
    legend_events = set()
    
    for _, row in df.iterrows():
        start_minutes = time_to_minutes(row['Start Time'])
        end_minutes = time_to_minutes(row['End Time'])
        y_pos = event_positions[row['Event']]
        
        # Handle cross-day events
        if end_minutes <= start_minutes:
            end_minutes += 1440
        
        event_color = get_event_color(row['Event'], color_map, username)
        show_legend = row['Event'] not in legend_events
        
        fig.add_trace(go.Scatter(
            x=[start_minutes, end_minutes, end_minutes, start_minutes, start_minutes],
            y=[y_pos - 0.4, y_pos - 0.4, y_pos + 0.4, y_pos + 0.4, y_pos - 0.4],
            fill='toself',
            fillcolor=event_color,
            line=dict(color=event_color, width=2),
            name=row['Event'],
            legendgroup=row['Event'],
            showlegend=show_legend,
            hovertemplate=f"<b>{row['Event']}</b><br>" +
                         f"时间: {row['Start Time']} - {row['End Time']}<br>" +
                         f"时长: {row['Duration (Hours)']:.1f} 小时<br>" +
                         f"备注: {row['Notes']}<extra></extra>",
        ))
        
        if show_legend:
            legend_events.add(row['Event'])
    
    # Set x-axis ticks
    x_ticks = list(range(0, 1441, 120))  # Every 2 hours
    x_labels = [minutes_to_time(x) for x in x_ticks]
    
    fig.update_layout(
        title="每日活动时间线" + (" (合并相同事件)" if merge_events else ""),
        xaxis_title="时间",
        yaxis_title="事件",
        xaxis=dict(
            tickmode='array',
            tickvals=x_ticks,
            ticktext=x_labels,
            range=[0, 1440]
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(unique_events))),
            ticktext=unique_events,
            range=[-0.5, len(unique_events) - 0.5]
        ),
        height=max(400, len(unique_events) * 60),
        hovermode='closest'
    )
    
    return fig

def create_pie_chart(df, color_map, username):
    """Create pie chart"""
    event_duration = df.groupby('Event')['Duration (Hours)'].sum().reset_index()
    colors = [get_event_color(event, color_map, username) for event in event_duration['Event']]
    
    fig = px.pie(
        event_duration, 
        values='Duration (Hours)', 
        names='Event',
        title="事件时间分配",
        color_discrete_sequence=colors
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate="<b>%{label}</b><br>" +
                     "时长: %{value:.1f}小时<br>" +
                     "百分比: %{percent}<extra></extra>"
    )
    
    return fig

def create_bar_chart(df, color_map, username):
    """Create bar chart"""
    event_duration = df.groupby('Event')['Duration (Hours)'].sum().reset_index()
    event_duration = event_duration.sort_values('Duration (Hours)', ascending=True)
    colors = [get_event_color(event, color_map, username) for event in event_duration['Event']]
    
    fig = px.bar(
        event_duration,
        x='Duration (Hours)',
        y='Event',
        orientation='h',
        title="事件时长统计",
        color_discrete_sequence=colors
    )
    
    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>" +
                     "时长: %{x:.1f}小时<extra></extra>"
    )
    
    fig.update_layout(height=400)
    
    return fig

def create_aggregated_timeline_chart(df, color_map, username):
    """Create aggregated timeline chart for date range analysis"""
    # Convert Date column to datetime and then to date string for consistency
    df_copy = df.copy()
    df_copy['Date'] = pd.to_datetime(df_copy['Date']).dt.date
    
    fig = go.Figure()
    
    # Get unique dates and sort them
    unique_dates = sorted(df_copy['Date'].unique())
    
    # Create y-axis positions for dates using string representation
    date_positions = {date: i for i, date in enumerate(unique_dates)}
    
    # Track which events have been added to legend
    legend_events = set()
    
    # Process each date separately and merge same events
    for date in unique_dates:
        date_df = df_copy[df_copy['Date'] == date].copy()
        # Merge same events for this date
        merged_df = merge_same_events(date_df)
        
        for _, row in merged_df.iterrows():
            start_minutes = time_to_minutes(row['Start Time'])
            end_minutes = time_to_minutes(row['End Time'])
            
            # If end time is smaller than start time, it's next day
            if end_minutes <= start_minutes:
                end_minutes += 1440
            
            y_pos = date_positions[date]
            event_color = get_event_color(row['Event'], color_map, username)
            
            # Check if this event should show in legend
            show_legend = row['Event'] not in legend_events
            if show_legend:
                legend_events.add(row['Event'])
            
            fig.add_trace(go.Scatter(
                x=[start_minutes, end_minutes],
                y=[y_pos, y_pos],
                mode='lines',
                line=dict(
                    color=event_color,
                    width=8
                ),
                name=row['Event'],
                legendgroup=row['Event'],
                showlegend=show_legend,
                hovertemplate=f"<b>{row['Event']}</b><br>" +
                             f"日期: {date}<br>" +
                             f"时间: {row['Start Time']} - {row['End Time']}<br>" +
                             f"时长: {row['Duration (Hours)']:.1f}小时<extra></extra>"
            ))
    
    # Set x-axis ticks
    x_ticks = list(range(0, 1441, 120))  # Every 2 hours
    x_labels = [minutes_to_time(x) for x in x_ticks]
    
    # Set y-axis labels
    y_labels = [str(date) for date in unique_dates]
    
    fig.update_layout(
        title="日期范围事件时间线 (合并相同事件)",
        xaxis_title="一天中的时间",
        yaxis_title="日期",
        xaxis=dict(
            tickmode='array',
            tickvals=x_ticks,
            ticktext=x_labels,
            range=[0, 1440]
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(unique_dates))),
            ticktext=y_labels
        ),
        height=max(400, len(unique_dates) * 40),
        hovermode='closest'
    )
    
    return fig

def create_daily_pattern_heatmap(df, color_map, username):
    """Create heatmap showing daily patterns"""
    # Convert time to hour buckets
    df_copy = df.copy()
    df_copy['Start Hour'] = df_copy['Start Time'].apply(lambda x: int(x.split(':')[0]))
    df_copy['Date'] = pd.to_datetime(df_copy['Date']).dt.date
    
    # Create pivot table for heatmap
    pivot_data = []
    for date in sorted(df_copy['Date'].unique()):
        date_data = df_copy[df_copy['Date'] == date]
        hour_events = {}
        
        for hour in range(24):
            events_in_hour = date_data[date_data['Start Hour'] == hour]['Event'].tolist()
            if events_in_hour:
                hour_events[hour] = ', '.join(set(events_in_hour))
            else:
                hour_events[hour] = ''
        
        hour_events['Date'] = str(date)
        pivot_data.append(hour_events)
    
    heatmap_df = pd.DataFrame(pivot_data)
    heatmap_df.set_index('Date', inplace=True)
    
    # Create binary matrix for visualization
    binary_matrix = heatmap_df.applymap(lambda x: 1 if x else 0)
    
    fig = go.Figure(data=go.Heatmap(
        z=binary_matrix.values,
        x=[f"{i:02d}:00" for i in range(24)],
        y=binary_matrix.index,
        colorscale='Blues',
        hovertemplate='日期: %{y}<br>时间: %{x}<br>事件: %{customdata}<extra></extra>',
        customdata=heatmap_df.values
    ))
    
    fig.update_layout(
        title="每日活动模式热力图",
        xaxis_title="一天中的小时",
        yaxis_title="日期",
        height=max(400, len(heatmap_df) * 30)
    )
    
    return fig

def main():
    # Check authentication first
    if not authenticate_user():
        return
    
    # Get current user info
    users = load_users()
    current_user = st.session_state.current_user
    user_info = users[current_user]
    
    # Update last login
    user_info['last_login'] = datetime.now().isoformat()
    users[current_user] = user_info
    save_users(users)
    
    # Main app header
    st.title(f"📊 个人行为追踪器 - 欢迎，{user_info['display_name']}！")
    
    # Logout button in sidebar
    st.sidebar.title("👤 用户信息")
    st.sidebar.write(f"**当前用户:** {user_info['display_name']}")
    st.sidebar.write(f"**用户名:** {current_user}")
    st.sidebar.write(f"**注册时间:** {user_info['created_at'][:10]}")
    
    if st.sidebar.button("退出登录"):
        st.session_state.authenticated = False
        st.session_state.current_user = None
        st.rerun()
    
    # Load user-specific data
    data = load_user_data(current_user)
    color_map = load_user_colors(current_user)
    
    # Navigation
    st.sidebar.title("📋 功能导航")
    page = st.sidebar.selectbox(
        "选择功能",
        ["📝 记录每日活动", "📊 数据可视化", "📚 历史管理", "🔍 搜索过滤", "👥 用户对比"]
    )
    
    if page == "📝 记录每日活动":
        st.header("记录每日活动")
        
        # Date selection
        selected_date = st.date_input("选择日期", datetime.now().date())
        date_str = selected_date.strftime("%Y-%m-%d")
        
        # Display existing events for the day
        if date_str in data:
            st.subheader("今日已记录事件")
            df = pd.DataFrame(data[date_str])
            # Rename columns for display
            df.columns = ['开始时间', '结束时间', '事件', '时长(小时)', '备注']
            st.dataframe(df, use_container_width=True)
        
        st.subheader("添加新事件")
        
        col1, col2 = st.columns(2)
        with col1:
            start_time_str = st.text_input(
                "开始时间", 
                placeholder="格式: HH:MM (如: 09:30)",
                help="请输入24小时制时间：小时:分钟"
            )
        with col2:
            end_time_str = st.text_input(
                "结束时间", 
                placeholder="格式: HH:MM (如: 17:30)",
                help="请输入24小时制时间：小时:分钟"
            )
        
        event_name = st.text_input("事件名称", placeholder="如：工作、学习、运动、娱乐...")
        notes = st.text_area("备注", placeholder="可选：添加详细描述...")
        
        if st.button("📌 添加事件", type="primary"):
            if event_name and start_time_str and end_time_str:
                # Validate time format
                if not validate_time(start_time_str):
                    st.error("❌ 开始时间格式无效。请按HH:MM格式输入")
                    return
                
                if not validate_time(end_time_str):
                    st.error("❌ 结束时间格式无效。请按HH:MM格式输入")
                    return
                
                # Calculate duration
                start_minutes = time_to_minutes(start_time_str)
                end_minutes = time_to_minutes(end_time_str)
                
                if end_minutes <= start_minutes:
                    end_minutes += 1440  # Handle cross-day events
                
                duration = (end_minutes - start_minutes) / 60
                
                # Prepare data
                new_event = {
                    "开始时间": start_time_str,
                    "结束时间": end_time_str,
                    "事件": event_name,
                    "时长(小时)": round(duration, 2),
                    "备注": notes
                }
                
                # Save data
                if date_str not in data:
                    data[date_str] = []
                
                data[date_str].append(new_event)
                save_user_data(current_user, data)
                
                # Ensure color mapping
                get_event_color(event_name, color_map, current_user)
                
                st.success(f"✅ 事件 '{event_name}' 已添加到 {date_str}")
                st.rerun()
            elif not event_name:
                st.error("❌ 请输入事件名称")
            elif not start_time_str or not end_time_str:
                st.error("❌ 请输入开始时间和结束时间")
    
    elif page == "📊 数据可视化":
        st.header("数据可视化")
        
        # Date selection
        selected_date = st.date_input("选择可视化日期", datetime.now().date())
        date_str = selected_date.strftime("%Y-%m-%d")
        
        if date_str in data and data[date_str]:
            df = pd.DataFrame(data[date_str])
            # Rename columns for display
            df.columns = ['Start Time', 'End Time', 'Event', 'Duration (Hours)', 'Notes']
            
            # Chart type selection
            chart_type = st.selectbox(
                "选择图表类型",
                ["甘特图 (时间线)", "饼图 (时间分配)", "柱状图 (时长统计)"]
            )
            
            if chart_type == "甘特图 (时间线)":
                # Option to merge same events
                merge_option = st.checkbox("合并连续相同事件", value=True, 
                                         help="将连续的相同事件时间段合并为单个条形")
                fig = create_gantt_chart(df, color_map, current_user, merge_events=merge_option)
                st.plotly_chart(fig, use_container_width=True)
            elif chart_type == "饼图 (时间分配)":
                fig = create_pie_chart(df, color_map, current_user)
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = create_bar_chart(df, color_map, current_user)
                st.plotly_chart(fig, use_container_width=True)
            
            # Statistics summary
            st.subheader("📈 统计摘要")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_events = len(df)
                st.metric("总事件数", total_events)
            
            with col2:
                total_hours = df['Duration (Hours)'].sum()
                st.metric("总时长", f"{total_hours:.1f}小时")
            
            with col3:
                unique_events = df['Event'].nunique()
                st.metric("不同事件类型", unique_events)
            
            # Detailed data table
            st.subheader("📋 详细数据")
            st.dataframe(df, use_container_width=True)
            
        else:
            st.info(f"📅 {date_str} 没有记录数据")
    
    elif page == "📚 历史管理":
        st.header("历史数据管理")
        
        if data:
            # Select date for editing
            available_dates = sorted(data.keys(), reverse=True)
            selected_date = st.selectbox("选择要管理的日期", available_dates)
            
            if selected_date:
                st.subheader(f"📅 {selected_date} 的数据")
                
                df = pd.DataFrame(data[selected_date])
                # Rename columns for display
                df.columns = ['开始时间', '结束时间', '事件', '时长(小时)', '备注']
                
                # Edit mode
                if st.checkbox("🔧 编辑模式"):
                    edited_df = st.data_editor(
                        df,
                        use_container_width=True,
                        num_rows="dynamic"
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("💾 保存更改", type="primary"):
                            # Keep Chinese column names for data consistency
                            data[selected_date] = edited_df.to_dict('records')
                            save_user_data(current_user, data)
                            st.success("✅ 数据保存成功")
                            st.rerun()
                    
                    with col2:
                        if st.button("🗑️ 删除这一天的数据", type="secondary"):
                            if st.session_state.get('confirm_delete', False):
                                del data[selected_date]
                                save_user_data(current_user, data)
                            st.success("✅ 数据删除成功")
                            st.session_state['confirm_delete'] = False
                            st.rerun()
                        else:
                             st.session_state['confirm_delete'] = True
                             st.warning("⚠️ 再次点击确认删除")
            else:
                st.dataframe(df, use_container_width=True)
                
            # Export data
            st.subheader("📤 数据导出")
            if st.button("下载CSV文件"):
                csv = df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                   label="📥 下载CSV",
                   data=csv,
                   file_name=f"{current_user}_data_{selected_date}.csv",
                   mime="text/csv"
                                            )
        else:
                                    st.info("📭 暂无历史数据")
    
    elif page == "🔍 搜索过滤":
        st.header("搜索与过滤")
        
        if data:
            # Create combined dataframe
            all_data = []
            for date, events in data.items():
                for event in events:
                    event_copy = event.copy()
                    event_copy['Date'] = date
                    all_data.append(event_copy)
            
            if all_data:
                df_all = pd.DataFrame(all_data)
                # Rename columns for display
                df_all.columns = ['开始时间', '结束时间', '事件', '时长(小时)', '备注', '日期']
                
                st.subheader("🔍 筛选条件")
                
                col1, col2 = st.columns(2)
                with col1:
                    # Date range filter
                    date_range = st.date_input(
                        "选择日期范围",
                        value=[pd.to_datetime(min(data.keys())).date(), 
                               pd.to_datetime(max(data.keys())).date()],
                        max_value=datetime.now().date()
                    )
                
                with col2:
                    # Event filter
                    unique_events = df_all['事件'].unique().tolist()
                    selected_events = st.multiselect(
                        "选择事件类型",
                        unique_events,
                        default=unique_events
                    )
                
                # Search text
                search_text = st.text_input("🔍 搜索关键词", placeholder="在事件名称或备注中搜索...")
                
                # Apply filters
                filtered_df = df_all.copy()
                
                # Date filter
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    filtered_df = filtered_df[
                        (pd.to_datetime(filtered_df['日期']).dt.date >= start_date) &
                        (pd.to_datetime(filtered_df['日期']).dt.date <= end_date)
                    ]
                
                # Event filter
                if selected_events:
                    filtered_df = filtered_df[filtered_df['事件'].isin(selected_events)]
                
                # Text search
                if search_text:
                    mask = (
                        filtered_df['事件'].str.contains(search_text, case=False, na=False) |
                        filtered_df['备注'].str.contains(search_text, case=False, na=False)
                    )
                    filtered_df = filtered_df[mask]
                
                # Display results
                st.subheader(f"📋 搜索结果 ({len(filtered_df)} 条记录)")
                
                if not filtered_df.empty:
                    # Sort by date and time
                    filtered_df = filtered_df.sort_values(['日期', '开始时间'])
                    st.dataframe(filtered_df, use_container_width=True)
                    
                    # Statistics
                    st.subheader("📊 统计信息")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("记录总数", len(filtered_df))
                    with col2:
                        st.metric("总时长", f"{filtered_df['时长(小时)'].sum():.1f}小时")
                    with col3:
                        st.metric("平均时长", f"{filtered_df['时长(小时)'].mean():.1f}小时")
                    with col4:
                        st.metric("涉及天数", filtered_df['日期'].nunique())
                    
                    # Time range visualization
                    if len(filtered_df) > 1:
                        st.subheader("📈 时间范围可视化")
                        
                        # Prepare data for visualization
                        vis_df = filtered_df.copy()
                        vis_df.columns = ['Start Time', 'End Time', 'Event', 'Duration (Hours)', 'Notes', 'Date']
                        
                        # Chart type selection
                        vis_type = st.selectbox(
                            "选择可视化类型",
                            ["聚合时间线", "每日模式热力图"]
                        )
                        
                        if vis_type == "聚合时间线":
                            fig = create_aggregated_timeline_chart(vis_df, color_map, current_user)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            fig = create_daily_pattern_heatmap(vis_df, color_map, current_user)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Export filtered data
                    st.subheader("📤 导出筛选结果")
                    if st.button("下载筛选结果CSV"):
                        csv = filtered_df.to_csv(index=False, encoding='utf-8-sig')
                        st.download_button(
                            label="📥 下载CSV",
                            data=csv,
                            file_name=f"{current_user}_filtered_data.csv",
                            mime="text/csv"
                        )
                else:
                    st.info("🔍 没有找到符合条件的记录")
        else:
            st.info("📭 暂无数据可供搜索")
    
    elif page == "👥 用户对比":
        st.header("用户对比分析")
        
        # Get all users
        all_users = load_users()
        user_list = list(all_users.keys())
        
        if len(user_list) > 1:
            st.subheader("📊 选择对比用户")
            
            # User selection
            other_users = [u for u in user_list if u != current_user]
            selected_user = st.selectbox("选择对比用户", other_users)
            
            if selected_user:
                # Date range selection
                date_range = st.date_input(
                    "选择对比日期范围",
                    value=[datetime.now().date() - timedelta(days=7), datetime.now().date()],
                    max_value=datetime.now().date()
                )
                
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    
                    # Load both users' data
                    current_user_data = load_user_data(current_user)
                    other_user_data = load_user_data(selected_user)
                    
                    # Filter data by date range
                    def filter_data_by_range(user_data, start_date, end_date):
                        filtered_data = []
                        for date_str, events in user_data.items():
                            date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
                            if start_date <= date_obj <= end_date:
                                for event in events:
                                    event_copy = event.copy()
                                    event_copy['Date'] = date_str
                                    filtered_data.append(event_copy)
                        return filtered_data
                    
                    current_filtered = filter_data_by_range(current_user_data, start_date, end_date)
                    other_filtered = filter_data_by_range(other_user_data, start_date, end_date)
                    
                    if current_filtered or other_filtered:
                        st.subheader("📈 对比分析")
                        
                        # Create comparison metrics
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**{all_users[current_user]['display_name']} (您)**")
                            
                            if current_filtered:
                                current_df = pd.DataFrame(current_filtered)
                                current_df.columns = ['开始时间', '结束时间', '事件', '时长(小时)', '备注', '日期']
                                
                                st.metric("总记录数", len(current_df))
                                st.metric("总时长", f"{current_df['时长(小时)'].sum():.1f}小时")
                                st.metric("不同事件类型", current_df['事件'].nunique())
                                st.metric("活跃天数", current_df['日期'].nunique())
                                
                                # Top events
                                top_events = current_df.groupby('事件')['时长(小时)'].sum().nlargest(3)
                                st.write("**主要活动:**")
                                for event, hours in top_events.items():
                                    st.write(f"• {event}: {hours:.1f}小时")
                            else:
                                st.info("该时间段内无数据")
                        
                        with col2:
                            st.write(f"**{all_users[selected_user]['display_name']}**")
                            
                            if other_filtered:
                                other_df = pd.DataFrame(other_filtered)
                                other_df.columns = ['开始时间', '结束时间', '事件', '时长(小时)', '备注', '日期']
                                
                                st.metric("总记录数", len(other_df))
                                st.metric("总时长", f"{other_df['时长(小时)'].sum():.1f}小时")
                                st.metric("不同事件类型", other_df['事件'].nunique())
                                st.metric("活跃天数", other_df['日期'].nunique())
                                
                                # Top events
                                top_events = other_df.groupby('事件')['时长(小时)'].sum().nlargest(3)
                                st.write("**主要活动:**")
                                for event, hours in top_events.items():
                                    st.write(f"• {event}: {hours:.1f}小时")
                            else:
                                st.info("该时间段内无数据")
                        
                        # Comparative visualization
                        if current_filtered and other_filtered:
                            st.subheader("📊 对比可视化")
                            
                            # Combine data for comparison
                            current_df['用户'] = all_users[current_user]['display_name']
                            other_df['用户'] = all_users[selected_user]['display_name']
                            
                            combined_df = pd.concat([current_df, other_df], ignore_index=True)
                            
                            # Event time comparison
                            event_comparison = combined_df.groupby(['事件', '用户'])['时长(小时)'].sum().reset_index()
                            
                            if not event_comparison.empty:
                                fig = px.bar(
                                    event_comparison,
                                    x='事件',
                                    y='时长(小时)',
                                    color='用户',
                                    title="事件时长对比",
                                    barmode='group'
                                )
                                fig.update_layout(height=400)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Daily activity comparison
                            daily_activity = combined_df.groupby(['日期', '用户'])['时长(小时)'].sum().reset_index()
                            
                            if not daily_activity.empty:
                                fig = px.line(
                                    daily_activity,
                                    x='日期',
                                    y='时长(小时)',
                                    color='用户',
                                    title="每日活动时长趋势",
                                    markers=True
                                )
                                fig.update_layout(height=400)
                                st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("选定时间范围内两位用户都没有数据")
        else:
            st.info("👥 系统中只有您一位用户，无法进行对比分析")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("📊 **多用户行为追踪器**")
    st.sidebar.markdown("版本 2.0")

if __name__ == "__main__":
    main()