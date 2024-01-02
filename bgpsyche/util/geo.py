from functools import reduce
import typing as t
import math

CountryUnknown = t.Literal['UNKNOWN']
COUNTRY_UNKNOWN = 'UNKNOWN'

Alpha2Official = t.Literal[
    'AD', 'AE', 'AF', 'AG', 'AI', 'AL', 'AM', 'AO', 'AQ', 'AR',
    'AS', 'AT', 'AU', 'AW', 'AX', 'AZ', 'BA', 'BB', 'BD', 'BE',
    'BF', 'BG', 'BH', 'BI', 'BJ', 'BL', 'BM', 'BN', 'BO', 'BQ',
    'BR', 'BS', 'BT', 'BV', 'BW', 'BY', 'BZ', 'CA', 'CC', 'CD',
    'CF', 'CG', 'CH', 'CI', 'CK', 'CL', 'CM', 'CN', 'CO', 'CR',
    'CU', 'CV', 'CW', 'CX', 'CY', 'CZ', 'DE', 'DJ', 'DK', 'DM',
    'DO', 'DZ', 'EC', 'EE', 'EG', 'EH', 'ER', 'ES', 'ET', 'FI',
    'FJ', 'FK', 'FM', 'FO', 'FR', 'GA', 'GB', 'GD', 'GE', 'GF',
    'GG', 'GH', 'GI', 'GL', 'GM', 'GN', 'GP', 'GQ', 'GR', 'GS',
    'GT', 'GU', 'GW', 'GY', 'HK', 'HM', 'HN', 'HR', 'HT', 'HU',
    'ID', 'IE', 'IL', 'IM', 'IN', 'IO', 'IQ', 'IR', 'IS', 'IT',
    'JE', 'JM', 'JO', 'JP', 'KE', 'KG', 'KH', 'KI', 'KM', 'KN',
    'KP', 'KR', 'KW', 'KY', 'KZ', 'LA', 'LB', 'LC', 'LI', 'LK',
    'LR', 'LS', 'LT', 'LU', 'LV', 'LY', 'MA', 'MC', 'MD', 'ME',
    'MF', 'MG', 'MH', 'MK', 'ML', 'MM', 'MN', 'MO', 'MP', 'MQ',
    'MR', 'MS', 'MT', 'MU', 'MV', 'MW', 'MX', 'MY', 'MZ', 'NA',
    'NC', 'NE', 'NF', 'NG', 'NI', 'NL', 'NO', 'NP', 'NR', 'NU',
    'NZ', 'OM', 'PA', 'PE', 'PF', 'PG', 'PH', 'PK', 'PL', 'PM',
    'PN', 'PR', 'PS', 'PT', 'PW', 'PY', 'QA', 'RE', 'RO', 'RS',
    'RU', 'RW', 'SA', 'SB', 'SC', 'SD', 'SE', 'SG', 'SH', 'SI',
    'SJ', 'SK', 'SL', 'SM', 'SN', 'SO', 'SR', 'SS', 'ST', 'SV',
    'SX', 'SY', 'SZ', 'TC', 'TD', 'TF', 'TG', 'TH', 'TJ', 'TK',
    'TL', 'TM', 'TN', 'TO', 'TR', 'TT', 'TV', 'TW', 'TZ', 'UA',
    'UG', 'UM', 'US', 'UY', 'UZ', 'VA', 'VC', 'VE', 'VG', 'VI',
    'VN', 'VU', 'WF', 'WS', 'YE', 'YT', 'ZA', 'ZM', 'ZW',
]
ALPHA2_OFFICIAL: t.Set[Alpha2Official] = \
    set(t.get_args(Alpha2Official))

Alpha2UserAssigned = t.Literal[
    'AA',
    'QM', 'QN', 'QO', 'QP', 'QQ', 'QR', 'QS', 'QT', 'QU', 'QV',
    'QW', 'QX', 'QY', 'QZ',
    'XA', 'XB', 'XC', 'XD', 'XE', 'XF', 'XG', 'XH', 'XI', 'XJ',
    'XK', 'XL', 'XM', 'XN', 'XO', 'XP', 'XQ', 'XR', 'XS', 'XT',
    'XU', 'XV', 'XW', 'XX', 'XY', 'XZ',
    'ZZ'
]
ALPHA2_USER_ASSIGNED: t.Set[Alpha2UserAssigned] = \
    set(t.get_args(Alpha2UserAssigned))

Alpha2ExceptionallyReserved = t.Literal[
    'AC', 'CP', 'CQ', 'DG', 'EA', 'EU', 'EZ', 'FX', 'IC', 'SU',
    'TA', 'UK', 'UN', 'AX', 'GG', 'IM', 'JE'
]
ALPHA2_EXCEPTIONALLY_RESERVED: t.Set[Alpha2ExceptionallyReserved] = \
    set(t.get_args(Alpha2ExceptionallyReserved))

Alpha2Transitional = t.Literal[
    'AN', 'BU', 'CS', 'NT', 'TP', 'YU', 'ZR',
]
ALPHA2_TRANSITIONAL: t.Set[Alpha2Transitional] = \
    set(t.get_args(Alpha2Transitional))

Alpha2UserAssignedCommon = t.Literal[
    'XK',
]
ALPHA2_USER_ASSIGNED_COMMON: t.Set[Alpha2UserAssignedCommon] = \
    set(t.get_args(Alpha2UserAssignedCommon))

Alpha2 = t.Union[
    Alpha2Official,
    Alpha2UserAssigned,
    Alpha2ExceptionallyReserved,
    Alpha2Transitional,
]

Alpha2WithLocation = t.Union[
    Alpha2Official,
    Alpha2ExceptionallyReserved,
    Alpha2Transitional,
    Alpha2UserAssignedCommon,
]
ALPHA2_WITH_LOCATION = reduce(set.union, [
    ALPHA2_OFFICIAL,
    ALPHA2_EXCEPTIONALLY_RESERVED,
    ALPHA2_TRANSITIONAL,
    ALPHA2_USER_ASSIGNED_COMMON,
])

class CountryInfo(t.TypedDict):
    name: str
    lat: float
    lon: float

COUNTRY_INFO: t.Dict[Alpha2WithLocation, CountryInfo] = {
    # official
    # ----------------------------------------------------------------------
    'AD': { 'name': 'Andorra', 'lat': 42.5, 'lon': 1.6 },
    'AE': { 'name': 'United Arab Emirates', 'lat': 24.0, 'lon': 54.0 },
    'AF': { 'name': 'Afghanistan', 'lat': 33.0, 'lon': 65.0 },
    'AG': { 'name': 'Antigua and Barbuda', 'lat': 17.05, 'lon': -61.8 },
    'AI': { 'name': 'Anguilla', 'lat': 18.25, 'lon': -63.1667 },
    'AL': { 'name': 'Albania', 'lat': 41.0, 'lon': 20.0 },
    'AM': { 'name': 'Armenia', 'lat': 40.0, 'lon': 45.0 },
    'AO': { 'name': 'Angola', 'lat': -12.5, 'lon': 18.5 },
    'AQ': { 'name': 'Antarctica', 'lat': -90.0, 'lon': 0.0 },
    'AR': { 'name': 'Argentina', 'lat': -34.0, 'lon': -64.0 },
    'AS': { 'name': 'American Samoa', 'lat': -14.3333, 'lon': -170.0 },
    'AT': { 'name': 'Austria', 'lat': 47.3333, 'lon': 13.3333 },
    'AU': { 'name': 'Australia', 'lat': -27.0, 'lon': 133.0 },
    'AW': { 'name': 'Aruba', 'lat': 12.5, 'lon': -69.9667 },
    'AX': { 'name': 'Åland Islands', 'lat': 60.12, 'lon': 19.9 },
    'AZ': { 'name': 'Azerbaijan', 'lat': 40.5, 'lon': 47.5 },
    'BA': { 'name': 'Bosnia and Herzegovina', 'lat': 44.0, 'lon': 18.0 },
    'BB': { 'name': 'Barbados', 'lat': 13.1667, 'lon': -59.5333 },
    'BD': { 'name': 'Bangladesh', 'lat': 24.0, 'lon': 90.0 },
    'BE': { 'name': 'Belgium', 'lat': 50.8333, 'lon': 4.0 },
    'BF': { 'name': 'Burkina Faso', 'lat': 13.0, 'lon': -2.0 },
    'BG': { 'name': 'Bulgaria', 'lat': 43.0, 'lon': 25.0 },
    'BH': { 'name': 'Bahrain', 'lat': 26.0, 'lon': 50.55 },
    'BI': { 'name': 'Burundi', 'lat': -3.5, 'lon': 30.0 },
    'BJ': { 'name': 'Benin', 'lat': 9.5, 'lon': 2.25 },
    'BL': { 'name': 'Saint Barthélemy', 'lat': 17.9,  'lon': -62.8 },
    'BM': { 'name': 'Bermuda', 'lat': 32.3333, 'lon': -64.75 },
    'BN': { 'name': 'Brunei', 'lat': 4.5, 'lon': 114.6667 },
    'BO': { 'name': 'Bolivia', 'lat': -17.0, 'lon': -65.0 },
    'BQ': { 'name': 'TODO', 'lat': 0, 'lon': 0 },
    'BR': { 'name': 'Brazil', 'lat': -10.0, 'lon': -55.0 },
    'BS': { 'name': 'Bahamas', 'lat': 24.25, 'lon': -76.0 },
    'BT': { 'name': 'Bhutan', 'lat': 27.5, 'lon': 90.5 },
    'BV': { 'name': 'Bouvet Island', 'lat': -54.4333, 'lon': 3.4 },
    'BW': { 'name': 'Botswana', 'lat': -22.0, 'lon': 24.0 },
    'BY': { 'name': 'Belarus', 'lat': 53.0, 'lon': 28.0 },
    'BZ': { 'name': 'Belize', 'lat': 17.25, 'lon': -88.75 },
    'CA': { 'name': 'Canada', 'lat': 60.0, 'lon': -95.0 },
    'CC': { 'name': 'Cocos (Keeling) Islands', 'lat': -12.5, 'lon': 96.8333 },
    'CD': { 'name': 'Congo, the Democratic Republic of the', 'lat': 0.0, 'lon': 25.0 },
    'CF': { 'name': 'Central African Republic', 'lat': 7.0, 'lon': 21.0 },
    'CG': { 'name': 'Congo', 'lat': -1.0, 'lon': 15.0 },
    'CH': { 'name': 'Switzerland', 'lat': 47.0, 'lon': 8.0 },
    'CI': { 'name': 'Ivory Coast', 'lat': 8.0, 'lon': -5.0 },
    'CK': { 'name': 'Cook Islands', 'lat': -21.2333, 'lon': -159.7667 },
    'CL': { 'name': 'Chile', 'lat': -30.0, 'lon': -71.0 },
    'CM': { 'name': 'Cameroon', 'lat': 6.0, 'lon': 12.0 },
    'CN': { 'name': 'China', 'lat': 35.0, 'lon': 105.0 },
    'CO': { 'name': 'Colombia', 'lat': 4.0, 'lon': -72.0 },
    'CR': { 'name': 'Costa Rica', 'lat': 10.0, 'lon': -84.0 },
    'CU': { 'name': 'Cuba', 'lat': 21.5, 'lon': -80.0 },
    'CV': { 'name': 'Cape Verde', 'lat': 16.0, 'lon': -24.0 },
    'CW': { 'name': 'Curaçao', 'lat': 12.12, 'lon': -68.9 },
    'CX': { 'name': 'Christmas Island', 'lat': -10.5, 'lon': 105.6667 },
    'CY': { 'name': 'Cyprus', 'lat': 35.0, 'lon': 33.0 },
    'CZ': { 'name': 'Czech Republic', 'lat': 49.75, 'lon': 15.5 },
    'DE': { 'name': 'Germany', 'lat': 51.0, 'lon': 9.0 },
    'DJ': { 'name': 'Djibouti', 'lat': 11.5, 'lon': 43.0 },
    'DK': { 'name': 'Denmark', 'lat': 56.0, 'lon': 10.0 },
    'DM': { 'name': 'Dominica', 'lat': 15.4167, 'lon': -61.3333 },
    'DO': { 'name': 'Dominican Republic', 'lat': 19.0, 'lon': -70.6667 },
    'DZ': { 'name': 'Algeria', 'lat': 28.0, 'lon': 3.0 },
    'EC': { 'name': 'Ecuador', 'lat': -2.0, 'lon': -77.5 },
    'EE': { 'name': 'Estonia', 'lat': 59.0, 'lon': 26.0 },
    'EG': { 'name': 'Egypt', 'lat': 27.0, 'lon': 30.0 },
    'EH': { 'name': 'Western Sahara', 'lat': 24.5, 'lon': -13.0 },
    'ER': { 'name': 'Eritrea', 'lat': 15.0, 'lon': 39.0 },
    'ES': { 'name': 'Spain', 'lat': 40.0, 'lon': -4.0 },
    'ET': { 'name': 'Ethiopia', 'lat': 8.0, 'lon': 38.0 },
    'FI': { 'name': 'Finland', 'lat': 64.0, 'lon': 26.0 },
    'FJ': { 'name': 'Fiji', 'lat': -18.0, 'lon': 175.0 },
    'FK': { 'name': 'Falkland Islands (Malvinas)', 'lat': -51.75, 'lon': -59.0 },
    'FM': { 'name': 'Micronesia, Federated States of', 'lat': 6.9167, 'lon': 158.25 },
    'FO': { 'name': 'Faroe Islands', 'lat': 62.0, 'lon': -7.0 },
    'FR': { 'name': 'France', 'lat': 46.0, 'lon': 2.0 },
    'GA': { 'name': 'Gabon', 'lat': -1.0, 'lon': 11.75 },
    'GB': { 'name': 'United Kingdom', 'lat': 54.0, 'lon': -2.0 },
    'GD': { 'name': 'Grenada', 'lat': 12.1167, 'lon': -61.6667 },
    'GE': { 'name': 'Georgia', 'lat': 42.0, 'lon': 43.5 },
    'GF': { 'name': 'French Guiana', 'lat': 4.0, 'lon': -53.0 },
    'GG': { 'name': 'Guernsey', 'lat': 49.5, 'lon': -2.56 },
    'GH': { 'name': 'Ghana', 'lat': 8.0, 'lon': -2.0 },
    'GI': { 'name': 'Gibraltar', 'lat': 36.1833, 'lon': -5.3667 },
    'GL': { 'name': 'Greenland', 'lat': 72.0, 'lon': -40.0 },
    'GM': { 'name': 'Gambia', 'lat': 13.4667, 'lon': -16.5667 },
    'GN': { 'name': 'Guinea', 'lat': 11.0, 'lon': -10.0 },
    'GP': { 'name': 'Guadeloupe', 'lat': 16.25, 'lon': -61.5833 },
    'GQ': { 'name': 'Equatorial Guinea', 'lat': 2.0, 'lon': 10.0 },
    'GR': { 'name': 'Greece', 'lat': 39.0, 'lon': 22.0 },
    'GS': { 'name': 'South Georgia and the South Sandwich Islands', 'lat': -54.5, 'lon': -37.0 },
    'GT': { 'name': 'Guatemala', 'lat': 15.5, 'lon': -90.25 },
    'GU': { 'name': 'Guam', 'lat': 13.4667, 'lon': 144.7833 },
    'GW': { 'name': 'Guinea-Bissau', 'lat': 12.0, 'lon': -15.0 },
    'GY': { 'name': 'Guyana', 'lat': 5.0, 'lon': -59.0 },
    'HK': { 'name': 'Hong Kong', 'lat': 22.25, 'lon': 114.1667 },
    'HM': { 'name': 'Heard Island and McDonald Islands', 'lat': -53.1, 'lon': 72.5167 },
    'HN': { 'name': 'Honduras', 'lat': 15.0, 'lon': -86.5 },
    'HR': { 'name': 'Croatia', 'lat': 45.1667, 'lon': 15.5 },
    'HT': { 'name': 'Haiti', 'lat': 19.0, 'lon': -72.4167 },
    'HU': { 'name': 'Hungary', 'lat': 47.0, 'lon': 20.0 },
    'ID': { 'name': 'Indonesia', 'lat': -5.0, 'lon': 120.0 },
    'IE': { 'name': 'Ireland', 'lat': 53.0, 'lon': -8.0 },
    'IL': { 'name': 'Israel', 'lat': 31.5, 'lon': 34.75 },
    'IM': { 'name': 'Isle of Man', 'lat': 54.23, 'lon': -4.55 },
    'IN': { 'name': 'India', 'lat': 20.0, 'lon': 77.0 },
    'IO': { 'name': 'British Indian Ocean Territory', 'lat': -6.0, 'lon': 71.5 },
    'IQ': { 'name': 'Iraq', 'lat': 33.0, 'lon': 44.0 },
    'IR': { 'name': 'Iran, Islamic Republic of', 'lat': 32.0, 'lon': 53.0 },
    'IS': { 'name': 'Iceland', 'lat': 65.0, 'lon': -18.0 },
    'IT': { 'name': 'Italy', 'lat': 42.8333, 'lon': 12.8333 },
    'JE': { 'name': 'Jersey', 'lat': 49.21, 'lon': -2.13 },
    'JM': { 'name': 'Jamaica', 'lat': 18.25, 'lon': -77.5 },
    'JO': { 'name': 'Jordan', 'lat': 31.0, 'lon': 36.0 },
    'JP': { 'name': 'Japan', 'lat': 36.0, 'lon': 138.0 },
    'KE': { 'name': 'Kenya', 'lat': 1.0, 'lon': 38.0 },
    'KG': { 'name': 'Kyrgyzstan', 'lat': 41.0, 'lon': 75.0 },
    'KH': { 'name': 'Cambodia', 'lat': 13.0, 'lon': 105.0 },
    'KI': { 'name': 'Kiribati', 'lat': 1.4167, 'lon': 173.0 },
    'KM': { 'name': 'Comoros', 'lat': -12.1667, 'lon': 44.25 },
    'KN': { 'name': 'Saint Kitts and Nevis', 'lat': 17.3333, 'lon': -62.75 },
    'KP': { 'name': 'Korea, Democratic People\'s Republic of', 'lat': 40.0, 'lon': 127.0 },
    'KR': { 'name': 'South Korea', 'lat': 37.0, 'lon': 127.5 },
    'KW': { 'name': 'Kuwait', 'lat': 29.3375, 'lon': 47.6581 },
    'KY': { 'name': 'Cayman Islands', 'lat': 19.5, 'lon': -80.5 },
    'KZ': { 'name': 'Kazakhstan', 'lat': 48.0, 'lon': 68.0 },
    'LA': { 'name': 'Lao People\'s Democratic Republic', 'lat': 18.0, 'lon': 105.0 },
    'LB': { 'name': 'Lebanon', 'lat': 33.8333, 'lon': 35.8333 },
    'LC': { 'name': 'Saint Lucia', 'lat': 13.8833, 'lon': -61.1333 },
    'LI': { 'name': 'Liechtenstein', 'lat': 47.1667, 'lon': 9.5333 },
    'LK': { 'name': 'Sri Lanka', 'lat': 7.0, 'lon': 81.0 },
    'LR': { 'name': 'Liberia', 'lat': 6.5, 'lon': -9.5 },
    'LS': { 'name': 'Lesotho', 'lat': -29.5, 'lon': 28.5 },
    'LT': { 'name': 'Lithuania', 'lat': 56.0, 'lon': 24.0 },
    'LU': { 'name': 'Luxembourg', 'lat': 49.75, 'lon': 6.1667 },
    'LV': { 'name': 'Latvia', 'lat': 57.0, 'lon': 25.0 },
    'LY': { 'name': 'Libya', 'lat': 25.0, 'lon': 17.0 },
    'MA': { 'name': 'Morocco', 'lat': 32.0, 'lon': -5.0 },
    'MC': { 'name': 'Monaco', 'lat': 43.7333, 'lon': 7.4 },
    'MD': { 'name': 'Moldova, Republic of', 'lat': 47.0, 'lon': 29.0 },
    'ME': { 'name': 'Montenegro', 'lat': 42.0, 'lon': 19.0 },
    'MF': { 'name': 'Saint Martin (French part)', 'lat': 18.08, 'lon': -63.05 },
    'MG': { 'name': 'Madagascar', 'lat': -20.0, 'lon': 47.0 },
    'MH': { 'name': 'Marshall Islands', 'lat': 9.0, 'lon': 168.0 },
    'MK': { 'name': 'Macedonia, the former Yugoslav Republic of', 'lat': 41.8333, 'lon': 22.0 },
    'ML': { 'name': 'Mali', 'lat': 17.0, 'lon': -4.0 },
    'MM': { 'name': 'Myanmar', 'lat': 22.0, 'lon': 98.0 },
    'MN': { 'name': 'Mongolia', 'lat': 46.0, 'lon': 105.0 },
    'MO': { 'name': 'Macao', 'lat': 22.1667, 'lon': 113.55 },
    'MP': { 'name': 'Northern Mariana Islands', 'lat': 15.2, 'lon': 145.75 },
    'MQ': { 'name': 'Martinique', 'lat': 14.6667, 'lon': -61.0 },
    'MR': { 'name': 'Mauritania', 'lat': 20.0, 'lon': -12.0 },
    'MS': { 'name': 'Montserrat', 'lat': 16.75, 'lon': -62.2 },
    'MT': { 'name': 'Malta', 'lat': 35.8333, 'lon': 14.5833 },
    'MU': { 'name': 'Mauritius', 'lat': -20.2833, 'lon': 57.55 },
    'MV': { 'name': 'Maldives', 'lat': 3.25, 'lon': 73.0 },
    'MW': { 'name': 'Malawi', 'lat': -13.5, 'lon': 34.0 },
    'MX': { 'name': 'Mexico', 'lat': 23.0, 'lon': -102.0 },
    'MY': { 'name': 'Malaysia', 'lat': 2.5, 'lon': 112.5 },
    'MZ': { 'name': 'Mozambique', 'lat': -18.25, 'lon': 35.0 },
    'NA': { 'name': 'Namibia', 'lat': -22.0, 'lon': 17.0 },
    'NC': { 'name': 'New Caledonia', 'lat': -21.5, 'lon': 165.5 },
    'NE': { 'name': 'Niger', 'lat': 16.0, 'lon': 8.0 },
    'NF': { 'name': 'Norfolk Island', 'lat': -29.0333, 'lon': 167.95 },
    'NG': { 'name': 'Nigeria', 'lat': 10.0, 'lon': 8.0 },
    'NI': { 'name': 'Nicaragua', 'lat': 13.0, 'lon': -85.0 },
    'NL': { 'name': 'Netherlands', 'lat': 52.5, 'lon': 5.75 },
    'NO': { 'name': 'Norway', 'lat': 62.0, 'lon': 10.0 },
    'NP': { 'name': 'Nepal', 'lat': 28.0, 'lon': 84.0 },
    'NR': { 'name': 'Nauru', 'lat': -0.5333, 'lon': 166.9167 },
    'NU': { 'name': 'Niue', 'lat': -19.0333, 'lon': -169.8667 },
    'NZ': { 'name': 'New Zealand', 'lat': -41.0, 'lon': 174.0 },
    'OM': { 'name': 'Oman', 'lat': 21.0, 'lon': 57.0 },
    'PA': { 'name': 'Panama', 'lat': 9.0, 'lon': -80.0 },
    'PE': { 'name': 'Peru', 'lat': -10.0, 'lon': -76.0 },
    'PF': { 'name': 'French Polynesia', 'lat': -15.0, 'lon': -140.0 },
    'PG': { 'name': 'Papua New Guinea', 'lat': -6.0, 'lon': 147.0 },
    'PH': { 'name': 'Philippines', 'lat': 13.0, 'lon': 122.0 },
    'PK': { 'name': 'Pakistan', 'lat': 30.0, 'lon': 70.0 },
    'PL': { 'name': 'Poland', 'lat': 52.0, 'lon': 20.0 },
    'PM': { 'name': 'Saint Pierre and Miquelon', 'lat': 46.8333, 'lon': -56.3333 },
    'PN': { 'name': 'Pitcairn', 'lat': -24.7, 'lon': -127.4 },
    'PR': { 'name': 'Puerto Rico', 'lat': 18.25, 'lon': -66.5 },
    'PS': { 'name': 'Palestinian Territory, Occupied', 'lat': 32.0, 'lon': 35.25 },
    'PT': { 'name': 'Portugal', 'lat': 39.5, 'lon': -8.0 },
    'PW': { 'name': 'Palau', 'lat': 7.5, 'lon': 134.5 },
    'PY': { 'name': 'Paraguay', 'lat': -23.0, 'lon': -58.0 },
    'QA': { 'name': 'Qatar', 'lat': 25.5, 'lon': 51.25 },
    'RE': { 'name': 'Réunion', 'lat': -21.1, 'lon': 55.6 },
    'RO': { 'name': 'Romania', 'lat': 46.0, 'lon': 25.0 },
    'RS': { 'name': 'Serbia', 'lat': 44.0, 'lon': 21.0 },
    'RU': { 'name': 'Russia', 'lat': 60.0, 'lon': 100.0 },
    'RW': { 'name': 'Rwanda', 'lat': -2.0, 'lon': 30.0 },
    'SA': { 'name': 'Saudi Arabia', 'lat': 25.0, 'lon': 45.0 },
    'SB': { 'name': 'Solomon Islands', 'lat': -8.0, 'lon': 159.0 },
    'SC': { 'name': 'Seychelles', 'lat': -4.5833, 'lon': 55.6667 },
    'SD': { 'name': 'Sudan', 'lat': 15.0, 'lon': 30.0 },
    'SE': { 'name': 'Sweden', 'lat': 62.0, 'lon': 15.0 },
    'SG': { 'name': 'Singapore', 'lat': 1.3667, 'lon': 103.8 },
    'SH': { 'name': 'Saint Helena, Ascension and Tristan da Cunha', 'lat': -15.9333, 'lon': -5.7 },
    'SI': { 'name': 'Slovenia', 'lat': 46.0, 'lon': 15.0 },
    'SJ': { 'name': 'Svalbard and Jan Mayen', 'lat': 78.0, 'lon': 20.0 },
    'SK': { 'name': 'Slovakia', 'lat': 48.6667, 'lon': 19.5 },
    'SL': { 'name': 'Sierra Leone', 'lat': 8.5, 'lon': -11.5 },
    'SM': { 'name': 'San Marino', 'lat': 43.7667, 'lon': 12.4167 },
    'SN': { 'name': 'Senegal', 'lat': 14.0, 'lon': -14.0 },
    'SO': { 'name': 'Somalia', 'lat': 10.0, 'lon': 49.0 },
    'SR': { 'name': 'Suriname', 'lat': 4.0, 'lon': -56.0 },
    'SS': { 'name': 'South Sudan', 'lat': 8.0, 'lon': 30.0 },
    'ST': { 'name': 'Sao Tome and Principe', 'lat': 1.0, 'lon': 7.0 },
    'SV': { 'name': 'El Salvador', 'lat': 13.8333, 'lon': -88.9167 },
    'SX': { 'name': 'Sint Maarten (Dutch part)', 'lat': 18.04, 'lon': -63.06 },
    'SY': { 'name': 'Syrian Arab Republic', 'lat': 35.0, 'lon': 38.0 },
    'SZ': { 'name': 'Swaziland', 'lat': -26.5, 'lon': 31.5 },
    'TC': { 'name': 'Turks and Caicos Islands', 'lat': 21.75, 'lon': -71.5833 },
    'TD': { 'name': 'Chad', 'lat': 15.0, 'lon': 19.0 },
    'TF': { 'name': 'French Southern Territories', 'lat': -43.0, 'lon': 67.0 },
    'TG': { 'name': 'Togo', 'lat': 8.0, 'lon': 1.1667 },
    'TH': { 'name': 'Thailand', 'lat': 15.0, 'lon': 100.0 },
    'TJ': { 'name': 'Tajikistan', 'lat': 39.0, 'lon': 71.0 },
    'TK': { 'name': 'Tokelau', 'lat': -9.0, 'lon': -172.0 },
    'TL': { 'name': 'Timor-Leste', 'lat': -8.55, 'lon': 125.5167 },
    'TM': { 'name': 'Turkmenistan', 'lat': 40.0, 'lon': 60.0 },
    'TN': { 'name': 'Tunisia', 'lat': 34.0, 'lon': 9.0 },
    'TO': { 'name': 'Tonga', 'lat': -20.0, 'lon': -175.0 },
    'TR': { 'name': 'Turkey', 'lat': 39.0, 'lon': 35.0 },
    'TT': { 'name': 'Trinidad and Tobago', 'lat': 11.0, 'lon': -61.0 },
    'TV': { 'name': 'Tuvalu', 'lat': -8.0, 'lon': 178.0 },
    'TW': { 'name': 'Taiwan', 'lat': 23.5, 'lon': 121.0 },
    'TZ': { 'name': 'Tanzania, United Republic of', 'lat': -6.0, 'lon': 35.0 },
    'UA': { 'name': 'Ukraine', 'lat': 49.0, 'lon': 32.0 },
    'UG': { 'name': 'Uganda', 'lat': 1.0, 'lon': 32.0 },
    'UM': { 'name': 'United States Minor Outlying Islands', 'lat': 19.2833, 'lon': 166.6 },
    'US': { 'name': 'United States', 'lat': 38.0, 'lon': -97.0 },
    'UY': { 'name': 'Uruguay', 'lat': -33.0, 'lon': -56.0 },
    'UZ': { 'name': 'Uzbekistan', 'lat': 41.0, 'lon': 64.0 },
    'VA': { 'name': 'Holy See (Vatican City State)', 'lat': 41.9, 'lon': 12.45 },
    'VC': { 'name': 'St. Vincent and the Grenadines', 'lat': 13.25, 'lon': -61.2 },
    'VE': { 'name': 'Venezuela', 'lat': 8.0, 'lon': -66.0 },
    'VG': { 'name': 'Virgin Islands, British', 'lat': 18.5, 'lon': -64.5 },
    'VI': { 'name': 'Virgin Islands, U.S.', 'lat': 18.3333, 'lon': -64.8333 },
    'VN': { 'name': 'Vietnam', 'lat': 16.0, 'lon': 106.0 },
    'VU': { 'name': 'Vanuatu', 'lat': -16.0, 'lon': 167.0 },
    'WF': { 'name': 'Wallis and Futuna', 'lat': -13.3, 'lon': -176.2 },
    'WS': { 'name': 'Samoa', 'lat': -13.5833, 'lon': -172.3333 },
    'YE': { 'name': 'Yemen', 'lat': 15.0, 'lon': 48.0 },
    'YT': { 'name': 'Mayotte', 'lat': -12.8333, 'lon': 45.1667 },
    'ZA': { 'name': 'South Africa', 'lat': -29.0, 'lon': 24.0 },
    'ZM': { 'name': 'Zambia', 'lat': -15.0, 'lon': 30.0 },
    'ZW': { 'name': 'Zimbabwe', 'lat': -20.0, 'lon': 30.0 },

    # transitional
    # ----------------------------------------------------------------------
    'AN': { 'name': 'Netherlands Antilles', 'lat': 12.25, 'lon': -68.75 },
    'BU': { 'name': 'Burma (Myanmar)', 'lat': 22.0, 'lon': 98.0 },
    'CS': { 'name': 'Czechoslovakia (now Montenegro and Serbia)', 'lat': 42.0, 'lon': 19.0 },
    'NT': { 'name': 'Neutral Zone (Divided between Iraq (IQ) and Saudi Arabia (SA))', 'lat': 33.0, 'lon': 44.0 },
    'TP': { 'name': 'East Timor (Timor-Leste)', 'lat': -8.55, 'lon': 125.5167 },
    'YU': { 'name': 'Yugoslavia (Macedonia)', 'lat': 41.8333, 'lon': 22.0 },
    'ZR': { 'name': 'Zaire (Democratic Republic of Congo)', 'lat': 0.0, 'lon': 25.0 },

    # exceptionally reserved
    # ----------------------------------------------------------------------
    'AC': { 'name': 'Ascension Island', 'lat': -7.93, 'lon': -14.42 },
    'CP': { 'name': 'Clipperton Island', 'lat': 10.3, 'lon': -109.22 },
    'CQ': { 'name': 'Island of Sark', 'lat': 49.43, 'lon': -2.37 },
    'DG': { 'name': 'Diego Garcia', 'lat': -7.31, 'lon': 72.41},
    'EA': { 'name': 'Ceuta, Melilla', 'lat': 35.3, 'lon': -2.95},
    # location: Gadheim, near Würzburg
    'EU': { 'name': 'European Union', 'lat': 49.84 , 'lon': 9.91 },
    # while Eurozone != EU, we only want rough estimates here, so this should be fine
    'EZ': { 'name': 'Eurozone', 'lat': 49.84 , 'lon': 9.91},
    'FX': { 'name': 'France, Metropolitan', 'lat': 46.0, 'lon': 2.0 },
    'IC': { 'name': 'Canary Islands', 'lat': 28, 'lon': -16 },
    # again, USSR != Russia, but good enough for us
    'SU': { 'name': 'USSR', 'lat': 60.0, 'lon': 100.0 },
    'TA': { 'name': 'Tristan da Cunha', 'lat': -37.06, 'lon': -12.31 },
    'UK': { 'name': 'United Kingdom', 'lat': 54.0, 'lon': -2.0 },
    # ... hopefully noone uses this for our calculations
    'UN': { 'name': 'United Nations', 'lat': 0, 'lon': 0 },

    # user assigned common
    # ----------------------------------------------------------------------
    'XK': { 'name': 'Kosovo', 'lat': 42.67, 'lon': 21.17 }
}


def harvesine_distance_km(
        origin: t.Tuple[float, float], destination: t.Tuple[float, float]
) -> float:
    """
    Calculate the Haversine distance. Note that this is up to 0.5% inaccurate as
    it assumes a sperical model of the earth. For our purposes, this should be
    good enough, as we only need the rough distane between countries.

    Credits:
    - https://stackoverflow.com/a/38187562
    - https://stackoverflow.com/a/19412565

    Examples
    --------
    >>> origin = (48.1372, 11.5756)  # Munich
    >>> destination = (52.5186, 13.4083)  # Berlin
    >>> round(distance(origin, destination), 1)
    504.2
    """
    R = 6373.0 # Approximate radius of earth in km

    lat1, lon1 = math.radians(origin[0]), math.radians(origin[1])
    lat2, lon2 = math.radians(destination[0]), math.radians(destination[1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = math.sin(dlat / 2)**2 + \
        math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    a = round(a, 6)
    assert 0 <= a <= 1, a
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def _compute_country_distances():
    out: t.Dict[
        Alpha2WithLocation, t.Dict[Alpha2WithLocation, float]
    ] = {}

    for c1, c1_info in COUNTRY_INFO.items():
        for c2, c2_info in COUNTRY_INFO.items():
            if c1 not in out: out[c1] = {}
            out[c1][c2] = harvesine_distance_km(
                (c1_info['lat'], c1_info['lon']),
                (c2_info['lat'], c2_info['lon']),
            )

    return out


COUNTRY_DISTANCES: t.Dict[
    Alpha2WithLocation,
    t.Dict[Alpha2WithLocation, float]
] = _compute_country_distances()


if __debug__:
    assert  600 < COUNTRY_DISTANCES['DE']['FR'] <  900
    assert 1000 < COUNTRY_DISTANCES['DE']['NO'] < 1300
    assert 4000 < COUNTRY_DISTANCES['DE']['RU'] < 6000
    assert 6000 < COUNTRY_DISTANCES['DE']['CN'] < 8000