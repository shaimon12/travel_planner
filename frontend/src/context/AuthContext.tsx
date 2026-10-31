import React, { createContext, useContext, useState, useEffect } from 'react';
import { User, AuthResponse } from '../types';
import { apiService } from '../services/api';

interface AuthContextType {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, name?: string, guestTripId?: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [token, setToken] = useState<string | null>(() => localStorage.getItem('travel_planner_token'));
  const [user, setUser] = useState<User | null>(() => {
    const saved = localStorage.getItem('travel_planner_user');
    return saved ? JSON.parse(saved) : null;
  });
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const initAuth = async () => {
      const storedToken = localStorage.getItem('travel_planner_token');
      if (storedToken) {
        try {
          const profile = await apiService.getMe();
          setUser(profile);
          localStorage.setItem('travel_planner_user', JSON.stringify(profile));
        } catch (err) {
          console.error('Session expired or invalid token:', err);
          logout();
        }
      }
      setIsLoading(false);
    };

    initAuth();

    const handleUnauthorized = () => {
      logout();
    };

    window.addEventListener('auth-unauthorized', handleUnauthorized);
    return () => window.removeEventListener('auth-unauthorized', handleUnauthorized);
  }, []);

  const handleAuthSuccess = (data: AuthResponse) => {
    setToken(data.access_token);
    setUser(data.user);
    localStorage.setItem('travel_planner_token', data.access_token);
    localStorage.setItem('travel_planner_user', JSON.stringify(data.user));
  };

  const login = async (email: string, password: string) => {
    const data = await apiService.login({ email, password });
    handleAuthSuccess(data);
  };

  const register = async (email: string, password: string, name?: string, guestTripId?: string) => {
    const data = await apiService.register({ email, password, name, guest_trip_id: guestTripId });
    handleAuthSuccess(data);
    localStorage.removeItem('guest_trip_id');
  };

  const logout = () => {
    setToken(null);
    setUser(null);
    localStorage.removeItem('travel_planner_token');
    localStorage.removeItem('travel_planner_user');
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        isAuthenticated: !!token,
        isLoading,
        login,
        register,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

