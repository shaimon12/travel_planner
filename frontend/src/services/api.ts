import axios from 'axios';
import {
  TripSession,
  TripDetail,
  TripCreateInput,
  ChatResponsePayload,
  DestinationAnalytics,
  User,
  AuthResponse,
} from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 45000, // accommodate multi-agent LLM reasoning
});

// Axios Request Interceptor: Attach JWT Bearer Token
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('travel_planner_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Axios Response Interceptor: Handle Unauthorized
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Don't clear on login/register attempt failures
      if (!error.config?.url?.includes('/auth/login') && !error.config?.url?.includes('/auth/register')) {
        localStorage.removeItem('travel_planner_token');
        localStorage.removeItem('travel_planner_user');
        window.dispatchEvent(new Event('auth-unauthorized'));
      }
    }
    return Promise.reject(error);
  }
);

export const apiService = {
  // ================= Auth Endpoints =================
  /**
   * Registers a new tenant user.
   */
  async register(data: { email: string; password: string; name?: string; guest_trip_id?: string }): Promise<AuthResponse> {
    const response = await apiClient.post<AuthResponse>('/auth/register', data);
    return response.data;
  },

  /**
   * Authenticates user and returns JWT token.
   */
  async login(data: { email: string; password: string }): Promise<AuthResponse> {
    const response = await apiClient.post<AuthResponse>('/auth/login', data);
    return response.data;
  },

  /**
   * Fetches current authenticated user profile.
   */
  async getMe(): Promise<User> {
    const response = await apiClient.get<User>('/auth/me');
    return response.data;
  },

  // ================= Trip Endpoints =================
  /**
   * Fetches list of all trip planning sessions for the authenticated user.
   */
  async fetchTrips(): Promise<TripSession[]> {
    const response = await apiClient.get<TripSession[]>('/trips');
    return response.data;
  },

  /**
   * Fetches single trip session details including full chat history and destination analytics.
   */
  async getTrip(tripId: string): Promise<TripDetail> {
    const response = await apiClient.get<TripDetail>(`/trips/${tripId}`);
    return response.data;
  },

  /**
   * Creates new trip session, triggering LangGraph 3-node agent workflow.
   */
  async createTrip(data: TripCreateInput): Promise<TripDetail> {
    const response = await apiClient.post<TripDetail>('/trips', data);
    return response.data;
  },

  /**
   * Sends user follow-up query to LangGraph conversational agent.
   */
  async sendMessage(tripId: string, message: string): Promise<ChatResponsePayload> {
    const response = await apiClient.post<ChatResponsePayload>(`/trips/${tripId}/chat`, {
      message,
    });
    return response.data;
  },

  /**
   * Deletes a trip session and cascades removal of chat history and analytics.
   */
  async deleteTrip(tripId: string): Promise<void> {
    await apiClient.delete(`/trips/${tripId}`);
  },

  /**
   * Fetches latest tourism analytics for a trip destination.
   */
  async getAnalytics(tripId: string): Promise<DestinationAnalytics> {
    const response = await apiClient.get<DestinationAnalytics>(`/trips/${tripId}/analytics`);
    return response.data;
  },
};
