export interface User {
  id: string;
  email: string;
  name?: string;
  created_at: string;
  updated_at: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export interface FlightOption {
  id?: string;
  airline: string;
  flight_number?: string;
  departure: string;
  arrival: string;
  duration?: string;
  stops?: string;
  price: string;
  link?: string;
  booking_url?: string;
  cabin?: string;
  tag?: string;
  origin?: string;
  destination?: string;
  date?: string;
}

export interface Message {
  id: string;
  trip_id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  flight_data?: FlightOption[];
  meta?: {
    type?: string;
    weather_report?: string;
    flight_data?: FlightOption[];
    [key: string]: any;
  };
  created_at: string;
}

export interface DestinationAnalytics {
  id: string;
  trip_id: string;
  destination: string;
  visitors?: string;
  growth?: string;
  satisfaction?: string;
  top_demographic?: string;
  best_month?: string;
  raw_data?: Record<string, any>;
  created_at: string;
  updated_at: string;
}

export interface TripSession {
  id: string;
  user_id?: string;
  is_guest?: boolean;
  title: string;
  destination?: string;
  raw_destination?: string;
  origin_city?: string;
  traveler_name?: string;
  start_date?: string;
  duration: number;
  budget: string;
  travel_type: string;
  group_size: number;
  children_ages?: string;
  interests: string[];
  dietary_restrictions: string[];
  status: string;
  created_at: string;
  updated_at: string;
}

export interface TripDetail extends TripSession {
  messages: Message[];
  analytics?: DestinationAnalytics;
}

export interface TripCreateInput {
  traveler_name: string;
  raw_destination: string;
  origin_city?: string;
  title?: string;
  start_date?: string;
  duration: number;
  budget: string;
  travel_type: string;
  group_size: number;
  children_ages?: string;
  interests: string[];
  dietary_restrictions: string[];
}

export interface ChatResponsePayload {
  response: string;
  content?: string;
  flight_data?: FlightOption[];
  message: Message;
  analytics?: DestinationAnalytics;
}

