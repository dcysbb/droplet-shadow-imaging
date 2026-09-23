// Geant4 单滴输运核心。阅读顺序建议：main → DetectorConstruction →
// Generator → DropletField → EventAction/SteppingAction。
// 坐标原点在液滴中心；电子从 z=-L1 出发，屏幕在 z=+L2。
// Python native.py 负责把 SI 数据导出为 CSV；此处读入后立即转成
// Geant4/CLHEP 内部单位，写回 CSV 时再显式除以 mm、MeV 等单位。
#include "G4Box.hh"
#include "G4DormandPrince745.hh"
#include "G4ElectroMagneticField.hh"
#include "G4EmStandardPhysics_option4.hh"
#include "G4EqMagElectricField.hh"
#include "G4Event.hh"
#include "G4FieldManager.hh"
#include "G4GeometryManager.hh"
#include "G4IntegrationDriver.hh"
#include "G4LogicalVolume.hh"
#include "G4NistManager.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4PVPlacement.hh"
#include "G4RunManager.hh"
#include "G4Sphere.hh"
#include "G4Step.hh"
#include "G4StepLimiterPhysics.hh"
#include "G4SystemOfUnits.hh"
#include "G4ThreeVector.hh"
#include "G4TransportationManager.hh"
#include "G4UserLimits.hh"
#include "G4VModularPhysicsList.hh"
#include "G4VUserActionInitialization.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4UserEventAction.hh"
#include "G4UserSteppingAction.hh"
#include "G4ChordFinder.hh"
#include "Randomize.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
// source.csv 每行一个初级电子；x/y 是束腰位置(mm)，xp/yp 是 dx/dz、dy/dz，
// energy 是动能(MeV)。这些字段仍是 CSV 原始数字，Generator 才乘单位。
struct Beam { double x, y, xp, yp, energy; };
// 读入后已转换成 Geant4 内部单位：r、φ、E_r。
struct Radial { double r, potential, field; };
// Python 已把球谐系数缩放为电势量纲；只保存 m>=0 的独立项。
struct Multipole { int ell, m; std::complex<double> coefficient; };

bool IsWaterVolume(const G4VPhysicalVolume* volume) {
  // 薄层保护几何把一滴水分成多个同材质体积；这些名字都应计作水。
  // field-* 体积在“无材料”对照中是几何标记，不产生水散射/能损。
  if (!volume) return false;
  const auto& name = volume->GetName();
  return name == "droplet" || name == "core" || name == "core-guard";
}

std::vector<Beam> ReadBeam(const std::string& path) {
  // 第一行是列名。只接受完整的 5 列，防止半截事件进入输运。
  std::ifstream file(path);
  if (!file) throw std::runtime_error("Cannot open source: " + path);
  std::string line;
  std::getline(file, line);
  std::vector<Beam> values;
  while (std::getline(file, line)) {
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream input(line);
    Beam beam{};
    if (input >> beam.x >> beam.y >> beam.xp >> beam.yp >> beam.energy) values.push_back(beam);
  }
  if (values.empty()) throw std::runtime_error("Source file has no primary electrons");
  return values;
}

std::vector<Radial> ReadRadial(const std::string& path) {
  // CSV 是 mm、V、V/mm；乘单位后 Geant4 才能正确积分 Lorentz 方程。
  std::ifstream file(path);
  if (!file) throw std::runtime_error("Cannot open field: " + path);
  std::string line;
  std::getline(file, line);
  std::vector<Radial> values;
  while (std::getline(file, line)) {
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream input(line);
    Radial item{};
    if (input >> item.r >> item.potential >> item.field)
      values.push_back({item.r * mm, item.potential * volt, item.field * volt / mm});
  }
  if (values.size() < 2) throw std::runtime_error("Field table has fewer than two samples");
  return values;
}

std::vector<Multipole> ReadMultipoles(const std::string& path) {
  // 系数只需乘 volt，不再乘长度：Python 导出时已按 R^(l+1) 缩放。
  std::ifstream file(path);
  if (!file) throw std::runtime_error("Cannot open multipoles: " + path);
  std::string line;
  std::getline(file, line);
  std::vector<Multipole> values;
  while (std::getline(file, line)) {
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream input(line);
    int ell, m;
    double re, im;
    if (input >> ell >> m >> re >> im) values.push_back({ell, m, {re * volt, im * volt}});
  }
  return values;
}

double AssociatedLegendre(int l, int m, double x) {
  // 递推求连带 Legendre 多项式，使用 Condon–Shortley 相位，
  // 与 Python/SciPy 的球谐函数约定一致，否则非对称场会翻转符号。
  double pmm = 1.0;
  const double root = std::sqrt(std::max(0.0, 1.0 - x * x));
  for (int i = 1; i <= m; ++i) pmm *= -(2 * i - 1) * root;
  if (l == m) return pmm;
  double pmmp1 = x * (2 * m + 1) * pmm;
  if (l == m + 1) return pmmp1;
  for (int ll = m + 2; ll <= l; ++ll) {
    double next = ((2 * ll - 1) * x * pmmp1 - (ll + m - 1) * pmm) / (ll - m);
    pmm = pmmp1;
    pmmp1 = next;
  }
  return pmmp1;
}

class DropletField final : public G4ElectroMagneticField {
  // Geant4 在每个积分子步调用 GetFieldValue；磁场分量始终为零。
  // 球内从径向表插值，球外用总净电荷的 1/r² 解析尾场，
  // 再加球谐非对称项，覆盖从束腰到探测器的整个传播路径。
 public:
  DropletField(std::vector<Radial> radial, std::vector<Multipole> multipoles)
      : radial_(std::move(radial)), multipoles_(std::move(multipoles)) {}

  // 静电场可交换电子动能与势能，不能让输运器按纯磁场处理。
  G4bool DoesFieldChangeEnergy() const override { return true; }

  void GetFieldValue(const G4double point[4], G4double value[6]) const override {
    const G4ThreeVector x(point[0], point[1], point[2]);
    const double r = x.mag();
    double er = 0.0;
    if (r >= radial_.back().r) {
      // 最后一个表点在液滴外半径 R；φ(R)·R/r² 就是外部 Coulomb 场。
      er = radial_.back().potential * radial_.back().r / (r * r);
    } else if (r > 0) {
      // 球内的 PB/薄壳场已由 Python 采样；只在相邻表点间线性插值。
      auto it = std::lower_bound(radial_.begin(), radial_.end(), r,
                                 [](const Radial& p, double q) { return p.r < q; });
      if (it == radial_.begin()) er = it->field;
      else {
        const auto& a = *(it - 1);
        const auto& b = *it;
        er = a.field + (b.field - a.field) * (r - a.r) / (b.r - a.r);
      }
    }
    G4ThreeVector electric = r > 0 ? x * (er / r) : G4ThreeVector();
    if (!multipoles_.empty()) {
      // 球谐势的笛卡尔梯度用小距离差分。贴近介电边界时，
      // 跨界面的中心差分会混合内外两套解，因此改用同侧单边差分。
      const double h = std::min(0.1 * um, std::max(0.02 * nm, r * 1e-5));
      const double radius = radial_.back().r;
      const double phi0 = MultipolePotential(x);
      for (int axis = 0; axis < 3; ++axis) {
        G4ThreeVector delta;
        delta[axis] = h;
        const auto plus = x + delta, minus = x - delta;
        const double phi_plus = MultipolePotential(plus);
        const double phi_minus = MultipolePotential(minus);
        double gradient = (phi_plus - phi_minus) / (2 * h);
        if ((plus.mag() < radius) != (r < radius) &&
            (minus.mag() < radius) == (r < radius)) gradient = (phi0 - phi_minus) / h;
        if ((minus.mag() < radius) != (r < radius) &&
            (plus.mag() < radius) == (r < radius)) gradient = (phi_plus - phi0) / h;
        electric[axis] -= gradient;
      }
    }
    // Geant4 数组约定是 (Bx,By,Bz,Ex,Ey,Ez)。
    value[0] = value[1] = value[2] = 0.0;
    value[3] = electric.x();
    value[4] = electric.y();
    value[5] = electric.z();
  }

 private:
  double MultipolePotential(const G4ThreeVector& x) const {
    // 实电势：m=0 项取实部，m>0 要把省略的 -m 共轭项乘 2。
    // 球内基函数 r^l，球外基函数 r^(-l-1)，在 r=R 接续。
    const double r = x.mag();
    if (r == 0) return 0.0;
    const double radius = radial_.back().r;
    const double cosine = x.z() / r;
    const double angle = std::atan2(x.y(), x.x());
    double result = 0.0;
    for (const auto& item : multipoles_) {
      const int l = item.ell, m = item.m;
      const double norm = std::sqrt((2.0*l+1.0)/(4.0*CLHEP::pi) *
                                    std::exp(std::lgamma(l-m+1.0)-std::lgamma(l+m+1.0)));
      const std::complex<double> y = norm * AssociatedLegendre(l, m, cosine) *
                                     std::polar(1.0, m * angle);
      const double rad = r < radius ? std::pow(r / radius, l) : std::pow(radius / r, l + 1);
      result += rad * (m == 0 ? (item.coefficient * y).real() :
                      2.0 * (item.coefficient * y).real());
    }
    return result;
  }
  std::vector<Radial> radial_;
  std::vector<Multipole> multipoles_;
};

class DetectorConstruction final : public G4VUserDetectorConstruction {
  // 这里的“Detector”是整个几何：大真空 world、多级近滴加密区、
  // 可选水材料球体和薄屏幕，不是 Python 的 MCP/荧光屏响应模型。
  // MCP 增益、PSF、像素与噪声都在 detector.py 后处理。
 public:
  DetectorConstruction(double radius_mm, double l1_mm, double l2_mm,
                       std::string material, double layer_nm, double layer_step_nm,
                       double world_step_mm, double far_step_mm,
                       double near_step_um, double core_step_um)
      : radius_(radius_mm * mm), l1_(l1_mm * mm), l2_(l2_mm * mm),
        material_(std::move(material)), layer_(layer_nm * nm), layer_step_(layer_step_nm * nm),
        world_step_(world_step_mm * mm), far_step_(far_step_mm * mm),
        near_step_(near_step_um * um), core_step_(core_step_um * um) {}

  G4VPhysicalVolume* Construct() override {
    auto* nist = G4NistManager::Instance();
    auto* vacuum = nist->FindOrBuildMaterial("G4_Galactic");
    auto* water = nist->FindOrBuildMaterial("G4_WATER");
    auto* world_solid = new G4Box("world", 100 * mm, 100 * mm,
                                   std::max(l1_, l2_) + 10 * mm);
    auto* world = new G4LogicalVolume(world_solid, vacuum, "world");
    // 远处 Coulomb 尾场虽弱，但若一步跨越几毫米，能量守恒会变差。
    // 因此用 20 mm、2 mm 同心分区逐级收紧最大步长。
    world->SetUserLimits(new G4UserLimits(world_step_));
    auto* physical = new G4PVPlacement(nullptr, G4ThreeVector(),
                                       world, "world", nullptr, false, 0);
    auto* far_zone = new G4LogicalVolume(new G4Sphere("far-zone", 0, 20 * mm,
                                                       0, CLHEP::twopi, 0, CLHEP::pi),
                                          vacuum, "far-zone");
    new G4PVPlacement(nullptr, {}, far_zone, "far-zone", world, false, 0);
    far_zone->SetUserLimits(new G4UserLimits(far_step_));
    auto* zone = new G4LogicalVolume(new G4Sphere("near-zone", 0, 2 * mm, 0, CLHEP::twopi, 0, CLHEP::pi),
                                     vacuum, "near-zone");
    new G4PVPlacement(nullptr, {}, zone, "near-zone", far_zone, false, 0);
    zone->SetUserLimits(new G4UserLimits(near_step_));
    auto* droplet_material = material_ == "water" ? water : vacuum;
    const std::string droplet_name = material_ == "water" ? "droplet" : "field-droplet";
    const std::string core_name = material_ == "water" ? "core" : "field-core";
    auto* drop = new G4LogicalVolume(new G4Sphere(droplet_name, 0, radius_, 0,
                                                   CLHEP::twopi, 0, CLHEP::pi),
                                     droplet_material, droplet_name);
    new G4PVPlacement(nullptr, {}, drop, droplet_name, zone, false, 0);
    if (layer_ > 0 && layer_ < radius_) {
      // 纳米界面层不能靠毫米级全空间网格来解析。薄壳本身限步，
      // 边界内外另放 2 nm 保护区，防止刚出壳的一大步跨过场尖峰。
      auto* outer_guard = new G4LogicalVolume(
          new G4Sphere("field-outer-guard", radius_, radius_ + 2 * nm,
                       0, CLHEP::twopi, 0, CLHEP::pi),
          vacuum, "field-outer-guard");
      new G4PVPlacement(nullptr, {}, outer_guard, "field-outer-guard", zone, false, 0);
      outer_guard->SetUserLimits(new G4UserLimits(layer_step_));
      auto* core = new G4LogicalVolume(new G4Sphere(core_name, 0, radius_ - layer_, 0,
                                                    CLHEP::twopi, 0, CLHEP::pi),
                                       droplet_material, core_name);
      new G4PVPlacement(nullptr, {}, core, core_name, drop, false, 0);
      const double inner_boundary = radius_ - layer_;
      if (inner_boundary > 2 * nm) {
        // 核心与薄层交界的内侧也需要保护；水与真空对照都保留几何。
        const std::string guard_name = material_ == "water" ? "core-guard" : "field-core-guard";
        auto* inner_guard = new G4LogicalVolume(
            new G4Sphere(guard_name, inner_boundary - 2 * nm, inner_boundary,
                         0, CLHEP::twopi, 0, CLHEP::pi),
            droplet_material, guard_name);
        new G4PVPlacement(nullptr, {}, inner_guard, guard_name, core, false, 0);
        inner_guard->SetUserLimits(new G4UserLimits(layer_step_));
      }
      drop->SetUserLimits(new G4UserLimits(layer_step_));
      core->SetUserLimits(new G4UserLimits(core_step_));
    } else {
      drop->SetUserLimits(new G4UserLimits(core_step_));
    }
    auto* screen = new G4LogicalVolume(new G4Box("screen", 40 * mm, 40 * mm, 0.5 * um),
                                       vacuum, "screen");
    // 薄屏幕只是记录穿过 z=L2 的粒子，不在 Geant4 内模拟 MCP。
    new G4PVPlacement(nullptr, G4ThreeVector(0, 0, l2_), screen, "screen", world, false, 0);
    return physical;
  }

 private:
  double radius_, l1_, l2_;
  std::string material_;
  double layer_, layer_step_, world_step_, far_step_, near_step_, core_step_;
};

class Generator final : public G4VUserPrimaryGeneratorAction {
  // 第 i 个 Geant4 event 读取 source.csv 第 i 行；保证单粒子记录
  // 可以靠 event_id 与原始相空间点一一对应。
 public:
  Generator(const std::vector<Beam>& beam, double l1) : beam_(beam), l1_(l1 * mm) {
    gun_ = std::make_unique<G4ParticleGun>(1);
    gun_->SetParticleDefinition(G4ParticleTable::GetParticleTable()->FindParticle("e-"));
  }
  void GeneratePrimaries(G4Event* event) override {
    const auto& item = beam_.at(event->GetEventID());
    gun_->SetParticlePosition({item.x * mm, item.y * mm, -l1_});
    gun_->SetParticleMomentumDirection(G4ThreeVector(item.xp, item.yp, 1).unit());
    gun_->SetParticleEnergy(item.energy * MeV);
    gun_->GeneratePrimaryVertex(event);
  }
 private:
  const std::vector<Beam>& beam_;
  double l1_;
  std::unique_ptr<G4ParticleGun> gun_;
};

class EventAction final : public G4UserEventAction {
  // 每个初级事件维护两个量：是否有初级电子触及水，以及事件对水滴
  // 的净电荷交换。后者仅用于评估静态电场假设是否会被曝光破坏。
 public:
  explicit EventAction(std::ofstream& deposition) : deposition_(deposition) {}
  void BeginOfEventAction(const G4Event*) override { entered_ = false; deposited_e_ = 0; }
  void EndOfEventAction(const G4Event* event) override {
    deposition_ << event->GetEventID() << ',' << deposited_e_ << '\n';
  }
  bool entered() const { return entered_; }
  void mark_entered() { entered_ = true; }
  void add_deposited(double charge_e) { deposited_e_ += charge_e; }
 private:
  bool entered_ = false;
  double deposited_e_ = 0;
  std::ofstream& deposition_;
};

class SteppingAction final : public G4UserSteppingAction {
  // 在每个 Geant4 步末观察材料边界和屏幕：
  // 1) 带电轨迹跨入/跨出水，更新沉积电荷；
  // 2) 出水粒子记入 exits.csv，避免屏幕孔径造成能损统计偏差；
  // 3) 电子到屏幕记入 hits.csv，然后停止追踪该粒子。
 public:
  SteppingAction(EventAction* event, std::ofstream& output, std::ofstream& exits)
      : event_(event), output_(output), exits_(exits) {}
  void UserSteppingAction(const G4Step* step) override {
    auto* track = step->GetTrack();
    const auto* pre = step->GetPreStepPoint();
    const auto* post = step->GetPostStepPoint();
    if (std::getenv("DROPLET_TRACE_FIRST") &&
        G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID() == 0 &&
        track->GetParentID() == 0) {
      std::cerr << std::setprecision(15);
      std::cerr << "TRACE," << pre->GetPosition().z()/mm << ','
                << pre->GetPosition().perp()/mm << ','
                << pre->GetKineticEnergy()/eV << ','
                << post->GetPosition().z()/mm << ','
                << post->GetKineticEnergy()/eV << '\n';
    }
    auto* pre_volume = pre->GetPhysicalVolume();
    auto* post_volume = post->GetPhysicalVolume();
    if (!pre_volume) return;
    const auto& name = pre_volume->GetName();
    const bool water_before = IsWaterVolume(pre_volume);
    const bool water_after = IsWaterVolume(post_volume);
    if (water_before != water_after) {
      // 进入水体加上粒子自身电荷，离开时减去；单位为 e。
      // 次级电子也会贡献，因此正值不简单等于初级束流数。
      event_->add_deposited(track->GetDefinition()->GetPDGCharge() * (water_after ? 1 : -1));
    }
    if (water_before && !water_after) {
      // 记录水滴出口的能量与方向；初级/次级由 parent_id 区分。
      const auto& position = post->GetPosition();
      exits_ << G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID() << ','
             << track->GetTrackID() << ',' << track->GetParentID() << ','
             << position.x()/mm << ',' << position.y()/mm << ',' << position.z()/mm << ','
             << post->GetKineticEnergy()/MeV << ','
             << post->GetMomentumDirection().x() << ','
             << post->GetMomentumDirection().y() << ','
             << post->GetMomentumDirection().z() << '\n';
    }
    if (track->GetParentID() == 0 && (water_before || water_after)) event_->mark_entered();
    if (name == "screen" && track->GetDefinition()->GetPDGEncoding() == 11) {
      // 只把电子（PDG 11）作为成像 hits；event_->entered() 标记
      // 该初级事件是否曾入水，不等同于屏幕上的这个次级自身穿滴。
      const auto& pos = pre->GetPosition();
      output_ << G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID() << ','
              << track->GetTrackID() << ',' << track->GetParentID() << ','
              << pos.x()/mm << ',' << pos.y()/mm << ',' << pre->GetKineticEnergy()/MeV << ','
              << track->GetMomentumDirection().x() << ',' << track->GetMomentumDirection().y() << ','
              << (event_->entered() ? 1 : 0) << '\n';
      track->SetTrackStatus(fStopAndKill);
    }
  }
 private:
  EventAction* event_;
  std::ofstream& output_;
  std::ofstream& exits_;
};

class Actions final : public G4VUserActionInitialization {
  // 把事件源、事件累计器和逐步观察器注册给 Geant4 RunManager。
 public:
  Actions(const std::vector<Beam>& beam, double l1, std::ofstream& output,
          std::ofstream& deposition, std::ofstream& exits)
      : beam_(beam), l1_(l1), output_(output), deposition_(deposition), exits_(exits) {}
  void Build() const override {
    SetUserAction(new Generator(beam_, l1_));
    auto* event = new EventAction(deposition_);
    SetUserAction(event);
    SetUserAction(new SteppingAction(event, output_, exits_));
  }
 private:
  const std::vector<Beam>& beam_;
  double l1_;
  std::ofstream& output_;
  std::ofstream& deposition_;
  std::ofstream& exits_;
};

class PhysicsList final : public G4VModularPhysicsList {
  // option4 包括精细电磁输运；StepLimiter 使几何区的用户限步生效。
  // cut_um 是次级产生阈值的“射程长度”，不是电子跟踪终止能量。
 public:
  explicit PhysicsList(double cut_um) {
    SetDefaultCutValue(cut_um * um);
    RegisterPhysics(new G4EmStandardPhysics_option4);
    RegisterPhysics(new G4StepLimiterPhysics);
  }
  void SetCuts() override { SetCutsWithDefault(); }
};
} // namespace

int main(int argc, char** argv) {
  // 参数顺序由 Python native.run_geant4 集中生成；不要直接修改
  // 此处位置参数而不同时更新 Python 适配器与端到端测试。
  try {
    if (argc != 18) {
      std::cerr << "Usage: droplet_g4 source.csv field.csv multipoles.csv hits.csv "
                   "radius_mm L1_mm L2_mm epsilon_water water|vacuum seed layer_nm layer_step_nm "
                   "world_step_mm far_step_mm near_step_um core_step_um cut_um\n";
      return 2;
    }
    const auto beam = ReadBeam(argv[1]);
    auto field = std::make_unique<DropletField>(ReadRadial(argv[2]), ReadMultipoles(argv[3]));
    std::ofstream output(argv[4]);
    if (!output) throw std::runtime_error("Cannot open hit output");
    std::ofstream deposition(std::string(argv[4]).substr(0, std::string(argv[4]).find_last_of('/')) + "/deposition.csv");
    if (!deposition) throw std::runtime_error("Cannot open deposition output");
    std::ofstream exits(std::string(argv[4]).substr(0, std::string(argv[4]).find_last_of('/')) + "/exits.csv");
    if (!exits) throw std::runtime_error("Cannot open exit output");
    deposition << "event_id,deposited_e\n";
    exits << "event_id,track_id,parent_id,x_mm,y_mm,z_mm,energy_MeV,ux,uy,uz\n";
    output << "event_id,track_id,parent_id,x_mm,y_mm,energy_MeV,ux,uy,entered\n";
    output << std::setprecision(15);
    exits << std::setprecision(15);
    const double radius = std::stod(argv[5]), l1 = std::stod(argv[6]), l2 = std::stod(argv[7]);
    const std::string material(argv[9]);
    if (material != "water" && material != "vacuum") throw std::runtime_error("Invalid material");
    CLHEP::HepRandom::setTheSeed(std::stol(argv[10]));
    auto manager = std::make_unique<G4RunManager>();
    manager->SetUserInitialization(new DetectorConstruction(radius, l1, l2, material,
                                                            std::stod(argv[11]), std::stod(argv[12]),
                                                            std::stod(argv[13]), std::stod(argv[14]),
                                                            std::stod(argv[15]), std::stod(argv[16])));
    manager->SetUserInitialization(new PhysicsList(std::stod(argv[17])));
    manager->SetUserInitialization(new Actions(beam, l1, output, deposition, exits));
    auto* fm = G4TransportationManager::GetTransportationManager()->GetFieldManager();
    // Dormand–Prince 自适应积分相对论带电粒子方程；用纳米级最小步
    // 和 0.01 μm 的弦/交点容差保护极薄电场区的落点与能量守恒。
    auto* equation = new G4EqMagElectricField(field.get());
    auto* stepper = new G4DormandPrince745(equation, 8);
    auto* driver = new G4IntegrationDriver<G4DormandPrince745>(0.05 * nm, stepper, 8);
    fm->SetDetectorField(field.get());
    fm->SetChordFinder(new G4ChordFinder(driver));
    fm->SetDeltaOneStep(0.01 * um);
    fm->SetDeltaIntersection(0.01 * um);
    manager->Initialize();
    manager->BeamOn(static_cast<int>(beam.size()));
    // 一行 source 对应一次 event；原始 CSV 留在输出目录供审计。
    output.close();
    deposition.close();
    exits.close();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "droplet_g4: " << error.what() << '\n';
    return 1;
  }
}
