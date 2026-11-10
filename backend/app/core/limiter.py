from slowapi import Limiter
from slowapi.util import get_remote_address

# IP-based rate limiter using client remote address to guard against bot abuse
limiter = Limiter(key_func=get_remote_address)

