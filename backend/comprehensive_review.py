"""
Comprehensive System Review
Test all components for functionality and errors
"""

import os

print('🔍 COMPREHENSIVE SYSTEM REVIEW')
print('=' * 60)
print()

# Test core imports
components = []

try:
    from app.config import Config  # noqa: F401
    components.append(('Config', 'SUCCESS', '✅'))
except Exception as e:
    components.append(('Config', f'FAILED - {e}', '❌'))

try:
    from app.services.binance_service import BinanceService
    components.append(('Binance Service', 'SUCCESS', '✅'))
except Exception as e:
    components.append(('Binance Service', f'FAILED - {e}', '❌'))

try:
    from app.services.database_service import db_service
    components.append(('Database Service', 'SUCCESS', '✅'))
except Exception as e:
    components.append(('Database Service', f'FAILED - {e}', '❌'))

try:
    from app.services.enhanced_ml_service import EnhancedMLService
    components.append(('Enhanced ML Service', 'SUCCESS', '✅'))
except Exception as e:
    components.append(('Enhanced ML Service', f'FAILED - {e}', '❌'))

try:
    from app.services.enhanced_trading_service import EnhancedTradingService
    components.append(('Enhanced Trading Service', 'SUCCESS', '✅'))
except Exception as e:
    components.append(('Enhanced Trading Service', f'FAILED - {e}', '❌'))

try:
    from app.main import app  # noqa: F401
    components.append(('FastAPI App', 'SUCCESS', '✅'))
except Exception as e:
    components.append(('FastAPI App', f'FAILED - {e}', '❌'))

# Print results
print('📊 CORE IMPORTS RESULTS:')
print('-' * 60)
for name, status, icon in components:
    print(f'{icon} {name:<25}: {status}')

print()
print('🔧 FUNCTIONALITY TESTS:')
print('-' * 60)

# Test database functionality
try:
    from app.services.database_service import db_service
    # Test basic database operations
    trades = db_service.get_all_trades(limit=1)
    print('✅ Database Query       : SUCCESS')
except Exception as e:
    print(f'❌ Database Query       : FAILED - {e}')

# Test Binance service basic functionality  
try:
    from app.services.binance_service import BinanceService
    binance = BinanceService()
    connectivity = binance.check_api_connectivity()
    if connectivity:
        print('✅ Binance API         : SUCCESS')
    else:
        print('⚠️  Binance API         : NO CONNECTION (check credentials)')
except Exception as e:
    print(f'❌ Binance API         : FAILED - {e}')

# Test enhanced ML service
try:
    from app.services.enhanced_ml_service import EnhancedMLService
    ml_service = EnhancedMLService()
    print('✅ ML Service Init     : SUCCESS')
except Exception as e:
    print(f'❌ ML Service Init     : FAILED - {e}')

# Test trading service integration
try:
    from app.services.enhanced_trading_service import EnhancedTradingService
    from app.services.binance_service import BinanceService
    from app.services.enhanced_ml_service import EnhancedMLService
    
    binance = BinanceService()
    ml_service = EnhancedMLService()
    trading_service = EnhancedTradingService(binance, ml_service)
    print('✅ Trading Service     : SUCCESS')
except Exception as e:
    print(f'❌ Trading Service     : FAILED - {e}')

print()
print('🧪 POSITION SIZING TEST:')
print('-' * 60)

try:
    # Run position sizing test
    from test_final_position_sizing import test_final_position_sizing
    test_final_position_sizing()
    print('✅ Position Sizing     : SUCCESS')
except Exception as e:
    print(f'❌ Position Sizing     : FAILED - {e}')

print()
print('📁 FILE STRUCTURE REVIEW:')
print('-' * 60)

required_files = [
    'app/config.py',
    'app/main.py', 
    'app/services/binance_service.py',
    'app/services/database_service.py',
    'app/services/enhanced_ml_service.py',
    'app/services/enhanced_trading_service.py',
    'app/models/database.py',
    'app/api/enhanced_routes_v2.py'
]

for file_path in required_files:
    if os.path.exists(file_path):
        print(f'✅ {file_path:<30}: EXISTS')
    else:
        print(f'❌ {file_path:<30}: MISSING')

print()
success_count = sum(1 for _, status, _ in components if 'SUCCESS' in status)
total_components = len(components)

print(f'📈 OVERALL HEALTH: {success_count}/{total_components} components working')
if success_count == total_components:
    print('🎉 ALL SYSTEMS OPERATIONAL!')
elif success_count >= total_components * 0.8:
    print('⚠️  MOSTLY OPERATIONAL - Minor issues detected')
else:
    print('❌ CRITICAL ISSUES - System needs attention')