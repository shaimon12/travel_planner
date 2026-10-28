import axios from 'axios';
import {
  TripSession,
  TripDetail,
  TripCreateInput,
  ChatResponsePayload,
  DestinationAnalytics,
} from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 45000, // accommodate multi-agent LLM reasoning
});

export const apiService = {
  /**
   * Fetches list of all trip planning sessions for sidebar history.
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
