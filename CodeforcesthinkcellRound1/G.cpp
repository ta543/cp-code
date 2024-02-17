// time-limit: 5000
// problem-url: https://codeforces.com/contest/1930/problem/G
// Prefix Max Set Counting

#include <algorithm>
#include <array>
#include <bitset>
#include <cassert>
#include <chrono>
#include <climits>
#include <cmath>
#include <complex>
#include <cstring>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <vector>
using namespace std;

using ll = long long;
using db = long double;
using str = string;

// pairs
using pi = pair<int, int>;
using pl = pair<ll, ll>;
using pd = pair<db, db>;
#define mp make_pair
#define f first
#define s second

#define tcT template <class T
#define tcTU tcT, class U

tcT > using V = vector<T>;
tcT, size_t SZ > using AR = array<T, SZ>;
using vi = V<int>;
using vb = V<bool>;
using vl = V<ll>;
using vd = V<db>;
using vs = V<str>;
using vpi = V<pi>;
using vpl = V<pl>;
using vpd = V<pd>;

// vectors
#define sz(X) int((X).size())
#define bg(X) begin(X)
#define all(X) bg(X), end(X)
#define rall(X) X.rbegin(), X.rend()
#define sor(X) sort(all(X))
#define rsz resize
#define ins insert
#define pb push_back
#define eb emplace_back
#define ft front()
#define bk back()

#define lb lower_bound
#define ub upper_bound
tcT > int lwb(V<T> &a, const T &b) { return int(lb(all(a), b) - bg(a)); }
tcT > int upb(V<T> &a, const T &b) { return int(ub(all(a), b) - bg(a)); }

// loops
#define FOR(i, a, b) for (int i = (a); i < (b); ++i)
#define F0R(i, a) FOR(i, 0, a)
#define ROF(i, a, b) for (int i = (b)-1; i >= (a); --i)
#define R0F(i, a) ROF(i, 0, a)
#define rep(a) F0R(_, a)
#define each(a, X) for (auto &a : X)

const int MOD = 998244353;  // 1e9+7;
const int MX = (int)2e5 + 5;
const ll BIG = 1e18;  // not too close to LLONG_MAX
const db PI = acos((db)-1);
const int dx[4]{1, 0, -1, 0}, dy[4]{0, 1, 0, -1};  // for every grid problem!!
mt19937 rng((uint32_t)chrono::steady_clock::now().time_since_epoch().count());
template <class T> using pqg = priority_queue<T, vector<T>, greater<T>>;

// bitwise ops
constexpr int pct(int X) { return __builtin_popcount(X); }  // # of bits set
constexpr int bits(int X) {

    return X == 0 ? 0 : 31 - __builtin_clz(X);
}
constexpr int p2(int X) { return 1 << X; }
constexpr int msk2(int X) { return p2(X) - 1; }

ll cdiv(ll a, ll b) {
    return a / b + ((a ^ b) > 0 && a % b);
}  // divide a by b rounded up
ll fdiv(ll a, ll b) {
    return a / b - ((a ^ b) < 0 && a % b);
}  // divide a by b rounded down

tcT > bool ckmin(T &a, const T &b) {
    return b < a ? a = b, 1 : 0;
}  // set a = min(a,b)
tcT > bool ckmax(T &a, const T &b) {
    return a < b ? a = b, 1 : 0;
}  // set a = max(a,b)

tcTU > T fstTrue(T lo, T hi, U f) {
    ++hi;
    assert(lo <= hi);  // assuming f is increasing
    while (lo < hi) {  // find first indeX such that f is true
        T mid = lo + (hi - lo) / 2;
        f(mid) ? hi = mid : lo = mid + 1;
    }
    return lo;
}
tcTU > T lstTrue(T lo, T hi, U f) {
    --lo;
    T d = 1;
    while (lo + d < hi) {
        if (!f(lo + d)) {
            hi = lo + d - 1;
            break;
        }
        d *= 2;
    }
    assert(lo <= hi);  // assuming f is decreasing
    while (lo < hi) {  // find first indeX such that f is true
        T mid = lo + (hi - lo + 1) / 2;
        f(mid) ? lo = mid : hi = mid - 1;
    }
    return lo;
}
tcT > void remDup(vector<T> &v) {  // sort and remove duplicates
    sort(all(v));
    v.erase(unique(all(v)), end(v));
}
tcTU > void safeErase(T &t, const U &u) {
    auto it = t.find(u);
    assert(it != end(t));
    t.erase(it);
}

inline namespace IO {
#define SFINAE(x, ...)                                                         \
    template <class, class = void> struct x : std::false_type {};              \
    template <class T> struct x<T, std::void_t<__VA_ARGS__>> : std::true_type {}

SFINAE(DefaultI, decltype(std::cin >> std::declval<T &>()));
SFINAE(DefaultO, decltype(std::cout << std::declval<T &>()));
SFINAE(IsTuple, typename std::tuple_size<T>::type);
SFINAE(Iterable, decltype(std::begin(std::declval<T>())));

template <auto &is> struct Reader {
    template <class T> void Impl(T &t) {
        if constexpr (DefaultI<T>::value) is >> t;
        else if constexpr (Iterable<T>::value) {
            for (auto &X : t) Impl(X);
        } else if constexpr (IsTuple<T>::value) {
            std::apply([this](auto &...args) { (Impl(args), ...); }, t);
        } else static_assert(IsTuple<T>::value, "No matching type for read");
    }
    template <class... Ts> void read(Ts &...ts) { ((Impl(ts)), ...); }
};

template <class... Ts> void re(Ts &...ts) { Reader<cin>{}.read(ts...); }
#define def(t, args...)                                                        \
    t args;                                                                    \
    re(args);

template <auto &os, bool debug, bool print_nd> struct Writer {
    string comma() const { return debug ? "," : ""; }
    template <class T> constexpr char Space(const T &) const {
        return print_nd && (Iterable<T>::value or IsTuple<T>::value) ? '\n'
                                                                     : ' ';
    }
    template <class T> void Impl(T const &t) const {
        if constexpr (DefaultO<T>::value) os << t;
        else if constexpr (Iterable<T>::value) {
            if (debug) os << '{';
            int i = 0;
            for (auto &&X : t)
                ((i++) ? (os << comma() << Space(X), Impl(X)) : Impl(X));
            if (debug) os << '}';
        } else if constexpr (IsTuple<T>::value) {
            if (debug) os << '(';
            std::apply(
                [this](auto const &...args) {
                    int i = 0;
                    (((i++) ? (os << comma() << " ", Impl(args)) : Impl(args)),
                     ...);
                },
                t);
            if (debug) os << ')';
        } else static_assert(IsTuple<T>::value, "No matching type for print");
    }
    template <class T> void ImplWrapper(T const &t) const {
        if (debug) os << "\033[0;31m";
        Impl(t);
        if (debug) os << "\033[0m";
    }
    template <class... Ts> void ps(Ts const &...ts) const {
        ((Impl(ts)), ...);
    }
    template <class F, class... Ts>
    void print_with_sep(const std::string &sep, F const &f,
                        Ts const &...ts) const {
        ImplWrapper(f), ((os << sep, ImplWrapper(ts)), ...), os << '\n';
    }
    void print_with_sep(const std::string &) const { os << '\n'; }
};

template <class... Ts> void pr(Ts const &...ts) {
    Writer<cout, false, true>{}.ps(ts...);
}
template <class... Ts> void ps(Ts const &...ts) {
    Writer<cout, false, true>{}.print_with_sep(" ", ts...);
}
}  // namespace IO

inline namespace Debug {
template <typename... Args> void err(Args... args) {
    Writer<cerr, true, false>{}.print_with_sep(" | ", args...);
}
template <typename... Args> void errn(Args... args) {
    Writer<cerr, true, true>{}.print_with_sep(" | ", args...);
}

void err_prefix(str func, int line, string args) {
    cerr << "\033[0;31m\u001b[1mDEBUG\033[0m"
         << " | "
         << "\u001b[34m" << func << "\033[0m"
         << ":"
         << "\u001b[34m" << line << "\033[0m"
         << " - "
         << "[" << args << "] = ";
}

#ifdef LOCAL
#define dbg(args...) err_prefix(__FUNCTION__, __LINE__, #args), err(args)
#define dbgn(args...) err_prefix(__FUNCTION__, __LINE__, #args), errn(args)
#else
#define dbg(...)
#define dbgn(args...)
#endif

const auto beg_time = std::chrono::high_resolution_clock::now();

double time_elapsed() {
    return chrono::duration<double>(std::chrono::high_resolution_clock::now() -
                                    beg_time)
        .count();
}
}  // namespace Debug

inline namespace FileIO {
void setIn(str s) { freopen(s.c_str(), "r", stdin); }
void setOut(str s) { freopen(s.c_str(), "w", stdout); }
void setIO(str s = "") {
    cin.tie(0)->sync_with_stdio(0);
    cout << fixed << setprecision(12);
    if (sz(s)) setIn(s + ".in"), setOut(s + ".out");
}
}

#define len(X) ll(X.size())
#define elif else if
 
#define eb emplace_back
#define stoi stoll
 
#define MIN(V) *min_element(all(v))
#define MAX(V) *max_element(all(v))
#define LB(C, X) distance((C).begin(), lower_bound(all(C), (X)))
#define UB(C, X) distance((C).begin(), upper_bound(all(C), (X)))
#define UNIQUE(X) \
  sort(all(X)), X.erase(unique(all(X)), X.end()), X.shrink_to_fit()
 
#define INT(...)   \
  int __VA_ARGS__; \
  read(__VA_ARGS__)
#define LL(...)   \
  ll __VA_ARGS__; \
  read(__VA_ARGS__)
#define STR(...)      \
  string __VA_ARGS__; \
  read(__VA_ARGS__)
#define CHAR(...)   \
  char __VA_ARGS__; \
  read(__VA_ARGS__)
#define DBL(...)      \
  double __VA_ARGS__; \
  read(__VA_ARGS__)
 
#define VEC(type, name, size) \
  vector<type> name(size);    \
  read(name)
#define VV(type, name, h, w)                     \
  vector<vector<type>> name(h, vector<type>(w)); \
  read(name)
 
void YES(bool tc = 1) { ps(tc ? "YES" : "NO"); }
void NO(bool tc = 1) { YES(!tc); }
void Yes(bool tc = 1) { ps(tc ? "Yes" : "No"); }
void No(bool tc = 1) { Yes(!tc); }
void yes(bool tc = 1) { ps(tc ? "yes" : "no"); }
void no(bool tc = 1) { yes(!tc); }

// START
 
/**
 * Description: modular arithmetic operations
 * Source:
 * KACTL
 * https://codeforces.com/blog/entry/63903
 * https://codeforces.com/contest/1261/submission/65632855 (tourist)
 * https://codeforces.com/contest/1264/submission/66344993 (ksun)
 * also see https://github.com/ecnerwala/cp-book/blob/master/src/modnum.hpp
 * (ecnerwal) Verification: https://open.kattis.com/problems/modulararithmetic
 */
 
template <int MOD, int RT> struct mint {
  static const int mod = MOD;
  static constexpr mint rt() { return RT; }  // primitive root for FFT
  int v;
  explicit operator int() const {
    return v;
  }  // explicit -> don't silently convert to int
  mint() : v(0) {}
  mint(ll _v) {
    v = int((-MOD < _v && _v < MOD) ? _v : _v % MOD);
    if (v < 0) v += MOD;
  }
  bool operator==(const mint &o) const { return v == o.v; }
  friend bool operator!=(const mint &a, const mint &b) { return !(a == b); }
  friend bool operator<(const mint &a, const mint &b) { return a.v < b.v; }
  friend istream &operator>>(istream &is, mint &a) {
    ll x;
    is >> x;
    a = mint(x);
    return is;
  }
  friend ostream &operator<<(ostream &os, mint a) {
    os << int(a);
    return os;
  }
 
  mint &operator+=(const mint &o) {
    if ((v += o.v) >= MOD) v -= MOD;
    return *this;
  }
  mint &operator-=(const mint &o) {
    if ((v -= o.v) < 0) v += MOD;
    return *this;
  }
  mint &operator*=(const mint &o) {
    v = int((ll)v * o.v % MOD);
    return *this;
  }
  mint &operator/=(const mint &o) { return (*this) *= inv(o); }
  friend mint pow(mint a, ll p) {
    mint ans = 1;
    assert(p >= 0);
    for (; p; p /= 2, a *= a)
      if (p & 1) ans *= a;
    return ans;
  }
  friend mint inv(const mint &a) {
    assert(a.v != 0);
    return pow(a, MOD - 2);
  }
 
  mint operator-() const { return mint(-v); }
  mint &operator++() { return *this += 1; }
  mint &operator--() { return *this -= 1; }
  friend mint operator+(mint a, const mint &b) { return a += b; }
  friend mint operator-(mint a, const mint &b) { return a -= b; }
  friend mint operator*(mint a, const mint &b) { return a *= b; }
  friend mint operator/(mint a, const mint &b) { return a /= b; }
};
 
using mi = mint<MOD, 5>;  // 5 is primitive root for both common mods
using vmi = V<mi>;
using pmi = pair<mi, mi>;
using vpmi = V<pmi>;
 
V<vmi> scmb;  // small combinations
void genComb(int SZ) {
  scmb.assign(SZ, vmi(SZ));
  scmb[0][0] = 1;
  FOR(i, 1, SZ)
  F0R(j, i + 1) scmb[i][j] = scmb[i - 1][j] + (j ? scmb[i - 1][j - 1] : 0);
}
 
/**
 * Author: Lukas Polacek
 * Date: 2009-10-30
 * License: CC0
 * Source: folklore/TopCoder
 * Description: Computes partial sums a[0] + a[1] + ... + a[pos - 1], and
 * updates single elements a[i], taking the difference between the old and new
 * value. Time: Both operations are $O(\log N)$. Status: Stress-tested
 */
 
tcT > struct BIT {
  int N;
  V<T> data;
  void init(int _N) {
    N = _N;
    data.rsz(N);
  }
  void add(int p, T x) {
    for (++p; p <= N; p += p & -p) data[p - 1] += x;
  }
  T sum(int l, int r) { return sum(r + 1) - sum(l); }
  T sum(int r) {
    T s = 0;
    for (; r; r -= r & -r) s += data[r - 1];
    return s;
  }
  int lower_bound(T sum) {
    if (sum <= 0) return -1;
    int pos = 0;
    for (int pw = 1 << 25; pw; pw >>= 1) {
      int npos = pos + pw;
      if (npos <= N && data[npos - 1] < sum)
        pos = npos, sum -= data[pos - 1];
    }
    return pos;
  }
};
 
/**
 * Description: wraps a lambda so it can call itself
 * Source: http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0200r0.html
 */
 
namespace std {
 
template <class Fun> class y_combinator_result {
  Fun fun_;
 
  public:
  template <class T>
  explicit y_combinator_result(T &&fun) : fun_(std::forward<T>(fun)) {}
 
  template <class... Args> decltype(auto) operator()(Args &&...args) {
    return fun_(std::ref(*this), std::forward<Args>(args)...);
  }
};
 
template <class Fun> decltype(auto) y_combinator(Fun &&fun) {
  return y_combinator_result<std::decay_t<Fun>>(std::forward<Fun>(fun));
}
 
}  // namespace std
 
void y_comb_demo() {
  cout << y_combinator([](auto gcd, int a, int b) -> int {
    return b == 0 ? a : gcd(b, a % b);
  })(20, 30)
       << "\n";  // outputs 10
}
 
void solve(int tc) {
  def(int, N);
  BIT<mi> B;
  B.init(N);
  V<vi> adj(N + 1);
  rep(N - 1) {
    def(int, u, v);
    adj.at(u).pb(v);
    adj.at(v).pb(u);
  }
  vi par(N + 1);
  vi st(N + 1), en(N + 1);
  int cnt = 0;
  auto dfs = y_combinator([&](auto dfs, int x) -> void {
    st[x] = cnt++;
    for (int y : adj[x])
      if (y != par[x]) {
        par[y] = x;
        dfs(y);
      }
    en[x] = cnt - 1;
  });
  dfs(1);
  assert(cnt == N);
  vi exists_below(N + 1);
  exists_below.at(1) = 1;
  auto sum_below = [&](int x) { return B.sum(st.at(x), en.at(x)); };
  auto zero_out = [&](int x) { B.add(x, -B.sum(x, x)); };
  map<int, int> present;
  ROF(v, 1, N + 1) {
    auto it = present.lb(st[v]);
    if (it != begin(present)) {
      int p = prev(it)->s;
      if (en[p] >= en[v]) continue;
    }
    int w = v;
    while (!exists_below[w]) {
      exists_below[w] = 1;
      w = par[w];
    }
    mi add = v == N ? 1 : sum_below(w);
    B.add(st[v], add);
    present[st[v]] = v;
    it = present.find(st[v]);
    while (next(it) != end(present)) {
      int q = next(it)->s;
      if (en[q] <= en[v]) {
        zero_out(st[q]);
        present.erase(next(it));
      } else break;
    }
  }
  ps(B.sum(0, N - 1));
}

int main() {
    setIO();
    int TC;
    re(TC);
    FOR(i, 1, TC + 1) solve(i);
}










