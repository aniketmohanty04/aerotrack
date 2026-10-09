import React from 'react';
import { AQIInfo } from '../types';
import {
  HeartPulse,
  Wind,
  Shield,
  Home,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  HelpCircle,
} from 'lucide-react';

interface RecommendationsProps {
  aqiValue: number | null;
  aqiInfo: AQIInfo;
}

interface ActionCard {
  title: string;
  advice: string;
  category: 'outdoor' | 'ventilation' | 'mask' | 'sensitive' | 'general';
  status: 'safe' | 'caution' | 'danger';
  icon: React.ReactNode;
}

export const Recommendations: React.FC<RecommendationsProps> = ({ aqiValue, aqiInfo }) => {
  const getGuidance = (aqi: number | null): ActionCard[] => {
    if (aqi === null) {
      return [
        {
          title: 'General Health',
          advice: 'Awaiting current air quality indices for this geographic coordinate.',
          category: 'general',
          status: 'caution',
          icon: <HelpCircle className="w-5 h-5" />,
        },
      ];
    }

    if (aqi <= 50) {
      return [
        {
          title: 'Outdoor Activities',
          advice: 'Enjoy outdoor activities. Ideal conditions for exercise, walking, and sports.',
          category: 'outdoor',
          status: 'safe',
          icon: <Wind className="w-5 h-5 text-emerald-400" />,
        },
        {
          title: 'Home Ventilation',
          advice: 'Open windows to let clean, fresh air circulate freely throughout the home.',
          category: 'ventilation',
          status: 'safe',
          icon: <Home className="w-5 h-5 text-emerald-400" />,
        },
        {
          title: 'Mask Usage',
          advice: 'No mask needed. Air quality is clean and poses minimal risk.',
          category: 'mask',
          status: 'safe',
          icon: <Shield className="w-5 h-5 text-emerald-400" />,
        },
        {
          title: 'Sensitive Groups',
          advice: 'Normal precautions. Safe for individuals with asthma, children, and older adults.',
          category: 'sensitive',
          status: 'safe',
          icon: <HeartPulse className="w-5 h-5 text-emerald-400" />,
        },
      ];
    } else if (aqi <= 100) {
      return [
        {
          title: 'Outdoor Activities',
          advice: 'Enjoy outdoor activities. Exceptionally sensitive people should monitor for symptoms.',
          category: 'outdoor',
          status: 'safe',
          icon: <Wind className="w-5 h-5 text-amber-400" />,
        },
        {
          title: 'Home Ventilation',
          advice: 'Open windows freely, especially outside of peak vehicular traffic hours.',
          category: 'ventilation',
          status: 'safe',
          icon: <Home className="w-5 h-5 text-amber-400" />,
        },
        {
          title: 'Mask Usage',
          advice: 'No mask needed for the general public; optional for allergy-prone individuals.',
          category: 'mask',
          status: 'safe',
          icon: <Shield className="w-5 h-5 text-amber-400" />,
        },
        {
          title: 'Sensitive Groups',
          advice: 'Normal precautions. Keep reliever medication accessible if unusually sensitive to ozone or dust.',
          category: 'sensitive',
          status: 'safe',
          icon: <HeartPulse className="w-5 h-5 text-amber-400" />,
        },
      ];
    } else if (aqi <= 150) {
      return [
        {
          title: 'Outdoor Activities',
          advice: 'Limit prolonged outdoor exertion. Sensitive groups should take frequent rest breaks.',
          category: 'outdoor',
          status: 'caution',
          icon: <Wind className="w-5 h-5 text-orange-400" />,
        },
        {
          title: 'Home Ventilation',
          advice: 'Consider closing windows during rush hours and dusty periods to maintain indoor air quality.',
          category: 'ventilation',
          status: 'caution',
          icon: <Home className="w-5 h-5 text-orange-400" />,
        },
        {
          title: 'Mask Usage',
          advice: 'Mask for sensitive groups. Well-fitted N95 masks helpful during prolonged outdoor exposure.',
          category: 'mask',
          status: 'caution',
          icon: <Shield className="w-5 h-5 text-orange-400" />,
        },
        {
          title: 'Sensitive Groups',
          advice: 'Sensitive groups take precautions. Children, elderly, and lung/heart patients should reduce heavy exertion.',
          category: 'sensitive',
          status: 'caution',
          icon: <HeartPulse className="w-5 h-5 text-orange-400" />,
        },
      ];
    } else if (aqi <= 200) {
      return [
        {
          title: 'Outdoor Activities',
          advice: 'Avoid prolonged outdoor exertion. Move fitness routines indoors.',
          category: 'outdoor',
          status: 'danger',
          icon: <Wind className="w-5 h-5 text-red-400" />,
        },
        {
          title: 'Home Ventilation',
          advice: 'Keep windows and doors securely closed. Run HEPA air purifiers inside.',
          category: 'ventilation',
          status: 'danger',
          icon: <Home className="w-5 h-5 text-red-400" />,
        },
        {
          title: 'Mask Usage',
          advice: 'N95 or KN95 respirators recommended for sensitive groups and anyone outdoors for extended periods.',
          category: 'mask',
          status: 'danger',
          icon: <Shield className="w-5 h-5 text-red-400" />,
        },
        {
          title: 'Sensitive Groups',
          advice: 'People with asthma or lung ailments must stay indoors in clean air spaces.',
          category: 'sensitive',
          status: 'danger',
          icon: <HeartPulse className="w-5 h-5 text-red-400" />,
        },
      ];
    } else if (aqi <= 300) {
      return [
        {
          title: 'Outdoor Activities',
          advice: 'Stay indoors. Avoid all outdoor physical activities and unnecessary trips.',
          category: 'outdoor',
          status: 'danger',
          icon: <Wind className="w-5 h-5 text-purple-400" />,
        },
        {
          title: 'Home Ventilation',
          advice: 'Seal windows and doors. Run purifiers continuously on medium/high speed.',
          category: 'ventilation',
          status: 'danger',
          icon: <Home className="w-5 h-5 text-purple-400" />,
        },
        {
          title: 'Mask Usage',
          advice: 'N95 or KN95 respirators strongly recommended for all individuals going outside.',
          category: 'mask',
          status: 'danger',
          icon: <Shield className="w-5 h-5 text-purple-400" />,
        },
        {
          title: 'Sensitive Groups',
          advice: 'Everyone stay indoors. Emergency conditions for vulnerable populations; avoid all exposure.',
          category: 'sensitive',
          status: 'danger',
          icon: <HeartPulse className="w-5 h-5 text-purple-400" />,
        },
      ];
    } else {
      return [
        {
          title: 'Outdoor Activities',
          advice: 'Stay indoors. Avoid all outdoor physical exertion and exposure entirely.',
          category: 'outdoor',
          status: 'danger',
          icon: <AlertTriangle className="w-5 h-5 text-rose-500" />,
        },
        {
          title: 'Home Ventilation',
          advice: 'Seal windows, run purifiers continuously. Do not ventilate with unpurified outdoor air.',
          category: 'ventilation',
          status: 'danger',
          icon: <Home className="w-5 h-5 text-rose-500" />,
        },
        {
          title: 'Mask Usage',
          advice: 'N95 or KN95 respirators strongly recommended for all individuals going outside.',
          category: 'mask',
          status: 'danger',
          icon: <Shield className="w-5 h-5 text-rose-500" />,
        },
        {
          title: 'Sensitive Groups',
          advice: 'Everyone stay indoors. Emergency health warning; seek medical aid if chest tightness develops.',
          category: 'sensitive',
          status: 'danger',
          icon: <HeartPulse className="w-5 h-5 text-rose-500" />,
        },
      ];
    }
  };

  const cards = getGuidance(aqiValue);

  const getStatusBadge = (card: ActionCard) => {
    // Category-specific badge labels and colors for danger/restriction states
    if (card.status === 'danger') {
      switch (card.category) {
        case 'ventilation':
          return (
            <span className="flex items-center gap-1 text-[11px] font-bold text-sky-400 bg-sky-500/10 border border-sky-500/30 px-2 py-0.5 rounded-full">
              <Home className="w-3 h-3" /> Adjust
            </span>
          );
        case 'mask':
          return (
            <span className="flex items-center gap-1 text-[11px] font-bold text-orange-400 bg-orange-500/10 border border-orange-500/30 px-2 py-0.5 rounded-full">
              <Shield className="w-3 h-3" /> Recommended
            </span>
          );
        case 'outdoor':
        case 'sensitive':
        default:
          return (
            <span className="flex items-center gap-1 text-[11px] font-bold text-rose-400 bg-rose-500/10 border border-rose-500/30 px-2 py-0.5 rounded-full">
              <XCircle className="w-3 h-3" /> Restrict
            </span>
          );
      }
    }

    if (card.status === 'caution') {
      return (
        <span className="flex items-center gap-1 text-[11px] font-bold text-amber-400 bg-amber-500/10 border border-amber-500/30 px-2 py-0.5 rounded-full">
          <AlertTriangle className="w-3 h-3" /> Caution
        </span>
      );
    }

    return (
      <span className="flex items-center gap-1 text-[11px] font-bold text-emerald-400 bg-emerald-500/10 border border-emerald-500/30 px-2 py-0.5 rounded-full">
        <CheckCircle2 className="w-3 h-3" /> Safe
      </span>
    );
  };

  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900/70 backdrop-blur-xl p-5 shadow-xl space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-cyan-500/20 text-cyan-400 border border-cyan-500/30">
            <HeartPulse className="w-4 h-4" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-white">Rule-Based Health Advisory</h2>
            <p className="text-xs text-slate-400">
              Personalized recommendations for AQI level: <span className="font-semibold text-slate-200">{aqiInfo.category}</span>
            </p>
          </div>
        </div>
      </div>

      <div
        className="grid gap-4 w-full"
        style={{
          gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
          maxWidth: '100%',
          overflow: 'hidden',
        }}
      >
        {cards.map((card, idx) => (
          <div
            key={idx}
            className="rounded-xl border border-slate-800 bg-slate-950/60 p-4 flex flex-col justify-between space-y-3 hover:border-slate-700 transition-colors"
          >
            <div className="flex items-start justify-between">
              <div className="p-2 rounded-xl bg-slate-900 border border-slate-800">
                {card.icon}
              </div>
              {getStatusBadge(card)}
            </div>

            <div className="space-y-1">
              <h3 className="text-sm font-bold text-white">{card.title}</h3>
              <p className="text-xs text-slate-300 leading-relaxed">
                {card.advice}
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
