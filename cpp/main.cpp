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
#ifdef G4MULTITHREADED
#include "G4MTRunManager.hh"
#endif
#include "G4AutoDelete.hh"
#include "G4Threading.hh"
#include "G4UserRunAction.hh"
#include "G4Run.hh"
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
#include <atomic>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
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

class DropletField final : public G4ElectroMagneticField {
  // Geant4 在每个积分子步调用 GetFieldValue；磁场分量始终为零。
  // 球内由径向电势插值求导，球外用总净电荷的 1/r² 解析尾场。
 public:
  explicit DropletField(const std::vector<Radial>& radial) : radial_(radial) {}

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
      // 同半径两行分别为内外侧极限。upper_bound 在边界处选外侧，
      // 并保证所在插值区间长度大于零；不跨电荷壳面进行平滑。
      auto it = std::upper_bound(radial_.begin(), radial_.end(), r,
                                 [](double q, const Radial& p) { return q < p.r; });
      if (it == radial_.begin()) er = it->field;
      else {
        const auto& a = *(it - 1);
        const auto& b = *it;
        const double width = b.r - a.r;
        const double t = (r - a.r) / width;
        const double slope = (b.potential - a.potential) / width;
        const double m0 = -a.field, m1 = -b.field;
        // φ 的三次 Hermite 插值；负导数给 E，保持两者自洽。
        er = -(m0 + (6*slope - 4*m0 - 2*m1)*t +
                    (-6*slope + 3*m0 + 3*m1)*t*t);
      }
    }
    G4ThreeVector electric = r > 0 ? x * (er / r) : G4ThreeVector();
    // Geant4 数组约定是 (Bx,By,Bz,Ex,Ey,Ez)。
    value[0] = value[1] = value[2] = 0.0;
    value[3] = electric.x();
    value[4] = electric.y();
    value[5] = electric.z();
  }

 private:
  const std::vector<Radial>& radial_; // 只读共享；生命周期覆盖全部 worker。
};

// 每个线程一套积分状态，只有原始场表共享。ChordFinder 拥有 driver，
// 自建 stepper/equation/field 则由这个包释放；不可让不同线程共用积分器。
struct FieldSetup {
  DropletField field;
  G4EqMagElectricField equation;
  G4DormandPrince745 stepper;
  std::unique_ptr<G4ChordFinder> chord;
  explicit FieldSetup(const std::vector<Radial>& radial)
      : field(radial), equation(&field), stepper(&equation, 8),
        chord(new G4ChordFinder(new G4IntegrationDriver<G4DormandPrince745>(
            0.05 * nm, &stepper, 8))) {
    auto* fm = G4TransportationManager::GetTransportationManager()->GetFieldManager();
    fm->SetDetectorField(&field);
    fm->SetChordFinder(chord.get());
    fm->SetDeltaOneStep(0.01 * um);
    fm->SetDeltaIntersection(0.01 * um);
  }
};

class DetectorConstruction final : public G4VUserDetectorConstruction {
  // 这里的“Detector”是整个几何：大真空 world、多级近滴加密区、
  // 可选水材料球体和薄屏幕，不是 Python 的 MCP/荧光屏响应模型。
  // MCP 增益、PSF、像素与噪声都在 detector.py 后处理。
 public:
  DetectorConstruction(double radius_mm, double l1_mm, double l2_mm,
                       std::string material, double layer_nm, double layer_step_nm,
                       double world_step_mm, double far_step_mm,
                       double near_step_um, double core_step_um,
                       const std::vector<Radial>& radial)
      : radius_(radius_mm * mm), l1_(l1_mm * mm), l2_(l2_mm * mm),
        material_(std::move(material)), layer_(layer_nm * nm), layer_step_(layer_step_nm * nm),
        world_step_(world_step_mm * mm), far_step_(far_step_mm * mm),
        near_step_(near_step_um * um), core_step_(core_step_um * um), radial_(radial) {}

  void ConstructSDandField() override {
    // 此回调在主线程/各 worker 内调用。G4AutoDelete 在对应线程退出时清理。
    G4AutoDelete::Register(new FieldSetup(radial_));
  }

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
  const std::vector<Radial>& radial_;
};

struct WorkerReport { int id; int events; };
struct OutputRegistry {
  std::string directory;
  std::mutex mutex;
  std::vector<WorkerReport> reports;
  bool show_progress = false;
  size_t total = 0;
  std::atomic<size_t> completed{0};
  std::mutex progress_mutex;
  std::chrono::steady_clock::time_point last_progress{};

  void ReportProgress(const char* phase, bool force = false) {
    if (!show_progress) return;
    // 非阻塞获取；只允许一个线程写快照，其余 worker 继续计算。
    std::unique_lock<std::mutex> lock(progress_mutex, std::try_to_lock);
    if (!lock.owns_lock()) return;
    const auto now = std::chrono::steady_clock::now();
    if (!force && now - last_progress < std::chrono::milliseconds(250)) return;
    last_progress = now;
    const auto path = directory + "/progress.json";
    std::ofstream out(path + ".tmp");
    out << "{\"completed\":" << completed.load(std::memory_order_relaxed)
        << ",\"total\":" << total << ",\"phase\":\"" << phase << "\"}\n";
    out.close();
    // 原子替换保证 Python 不读到半行 JSON。进度 IO 失败不终止物理输运。
    if (out) {
      std::error_code error;
      std::filesystem::rename(path + ".tmp", path, error);
    }
  }
  void CompleteEvent() {
    if (!show_progress) return;
    const size_t done = completed.fetch_add(1, std::memory_order_relaxed) + 1;
    // 按已结束的完整事件计数，不用 event_id（多线程会乱序完成）。
    // 仅在少量事件边界尝试刷新，绝不在粒子积分的每个 step 中刷新。
    const size_t interval = std::max<size_t>(1, std::min<size_t>(1000, total / 1000));
    if (done % interval == 0 || done == total) ReportProgress("transport", done == total);
  }
};

// 文件按 worker 分片，流本身使用缓冲。锁仅用于每个 worker 结束时登记，
// 不在每个 step/hit 上竞争锁。空 worker 也写表头，便于统一读取。
struct WorkerOutput {
  std::ofstream hits, exits, deposition;
  int id = G4Threading::G4GetThreadId();
  int events = 0;
  explicit WorkerOutput(const std::string& directory) {
    const auto prefix = directory + "/worker_" + std::to_string(id);
    hits.open(prefix + "_hits.csv");
    exits.open(prefix + "_exits.csv");
    deposition.open(prefix + "_deposition.csv");
    if (!hits || !exits || !deposition) throw std::runtime_error("Cannot open worker output");
    hits << "event_id,track_id,parent_id,x_mm,y_mm,energy_MeV,ux,uy,entered\n";
    exits << "event_id,track_id,parent_id,x_mm,y_mm,z_mm,energy_MeV,ux,uy,uz\n";
    deposition << "event_id,deposited_e\n";
    hits << std::setprecision(15);
    exits << std::setprecision(15);
    deposition << std::setprecision(15);
  }
  void Close() {
    hits.close(); exits.close(); deposition.close();
    if (!hits || !exits || !deposition) throw std::runtime_error("Worker output write failed");
  }
};

class WorkerRunAction final : public G4UserRunAction {
 public:
  WorkerRunAction(std::shared_ptr<WorkerOutput> output, OutputRegistry& registry)
      : output_(std::move(output)), registry_(registry) {}
  void EndOfRunAction(const G4Run* run) override {
    if (run->GetNumberOfEvent() != output_->events)
      throw std::runtime_error("Worker event count mismatch");
    output_->Close();
    std::lock_guard<std::mutex> lock(registry_.mutex);
    registry_.reports.push_back({output_->id, output_->events});
  }
 private:
  std::shared_ptr<WorkerOutput> output_;
  OutputRegistry& registry_;
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
  EventAction(std::shared_ptr<WorkerOutput> output, OutputRegistry& registry)
      : output_(std::move(output)), registry_(registry) {}
  void BeginOfEventAction(const G4Event*) override { entered_ = false; deposited_e_ = 0; }
  void EndOfEventAction(const G4Event* event) override {
    if (event->IsAborted()) throw std::runtime_error("Aborted primary event");
    output_->deposition << event->GetEventID() << ',' << deposited_e_ << '\n';
    ++output_->events;
    registry_.CompleteEvent();
  }
  bool entered() const { return entered_; }
  void mark_entered() { entered_ = true; }
  void add_deposited(double charge_e) { deposited_e_ += charge_e; }
 private:
  bool entered_ = false;
  double deposited_e_ = 0;
  std::shared_ptr<WorkerOutput> output_;
  OutputRegistry& registry_;
};

class SteppingAction final : public G4UserSteppingAction {
  // 在每个 Geant4 步末观察材料边界和屏幕：
  // 1) 带电轨迹跨入/跨出水，更新沉积电荷；
  // 2) 出水粒子记入 exits.csv，避免屏幕孔径造成能损统计偏差；
  // 3) 电子到屏幕记入 hits.csv，然后停止追踪该粒子。
 public:
  SteppingAction(EventAction* event, std::ofstream& output, std::ofstream& exits)
      : event_(event), output_(output), exits_(exits),
        trace_first_(std::getenv("DROPLET_TRACE_FIRST") != nullptr) {}
  void UserSteppingAction(const G4Step* step) override {
    auto* track = step->GetTrack();
    const auto* pre = step->GetPreStepPoint();
    const auto* post = step->GetPostStepPoint();
    // 调试开关在启动时固定。不要在数百万 step 中反复查询进程环境：
    // 某些 libc 的 getenv 带锁，会让本应独立的 worker 发生竞争。
    if (trace_first_ &&
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
  const bool trace_first_;
};

class Actions final : public G4VUserActionInitialization {
  // 把事件源、事件累计器和逐步观察器注册给 Geant4 RunManager。
 public:
  Actions(const std::vector<Beam>& beam, double l1, OutputRegistry& registry)
      : beam_(beam), l1_(l1), registry_(registry) {}
  void BuildForMaster() const override {} // 主线程不产生粒子、不打开输出文件。
  void Build() const override {
    auto output = std::make_shared<WorkerOutput>(registry_.directory);
    SetUserAction(new WorkerRunAction(output, registry_));
    SetUserAction(new Generator(beam_, l1_));
    auto* event = new EventAction(output, registry_);
    SetUserAction(event);
    SetUserAction(new SteppingAction(event, output->hits, output->exits));
  }
 private:
  const std::vector<Beam>& beam_;
  double l1_;
  OutputRegistry& registry_;
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
    if (argc == 2 && std::string(argv[1]) == "--capabilities") {
      std::cout << "{\"protocol\":2,\"progress\":true,\"multithreaded\":"
#ifdef G4MULTITHREADED
                << "true"
#else
                << "false"
#endif
                << "}\n";
      return 0;
    }
    if (argc != 17) {
      std::cerr << "Usage: droplet_g4 source.csv field.csv output_directory "
                   "radius_mm L1_mm L2_mm water|vacuum seed layer_nm layer_step_nm "
                   "world_step_mm far_step_mm near_step_um core_step_um cut_um threads\n";
      return 2;
    }
    const auto beam = ReadBeam(argv[1]);
    const auto radial = ReadRadial(argv[2]);
    OutputRegistry registry;
    registry.directory = argv[3];
    std::filesystem::create_directories(registry.directory);
    const auto* progress_flag = std::getenv("DROPLET_SHOW_PROGRESS");
    registry.show_progress = progress_flag && std::string(progress_flag) == "1";
    registry.total = beam.size();
    registry.ReportProgress("initializing", true);
    const double radius = std::stod(argv[4]), l1 = std::stod(argv[5]), l2 = std::stod(argv[6]);
    const std::string material(argv[7]);
    if (material != "water" && material != "vacuum") throw std::runtime_error("Invalid material");
    const int threads = std::stoi(argv[16]);
    if (threads < 1) throw std::runtime_error("threads must be positive");
    // Python 已解析 0=auto，C++ 接收实际线程数。禁止外部变量悄悄覆盖。
    if (std::getenv("G4FORCENUMBEROFTHREADS"))
      throw std::runtime_error("Unset G4FORCENUMBEROFTHREADS; use geant4_threads instead");
    CLHEP::HepRandom::setTheSeed(std::stol(argv[8]));
    std::unique_ptr<G4RunManager> manager;
    if (threads == 1) manager = std::make_unique<G4RunManager>();
    else {
#ifdef G4MULTITHREADED
      auto mt = std::make_unique<G4MTRunManager>();
      mt->SetNumberOfThreads(threads);
      // 默认逐事件分配种子；不自行按 worker 复制或重置随机流。
      G4MTRunManager::SetSeedOncePerCommunication(0);
      manager = std::move(mt);
#else
      throw std::runtime_error("Geant4 was built without multithreading support");
#endif
    }
    manager->SetUserInitialization(new DetectorConstruction(radius, l1, l2, material,
                                                            std::stod(argv[9]), std::stod(argv[10]),
                                                            std::stod(argv[11]), std::stod(argv[12]),
                                                            std::stod(argv[13]), std::stod(argv[14]), radial));
    manager->SetUserInitialization(new PhysicsList(std::stod(argv[15])));
    manager->SetUserInitialization(new Actions(beam, l1, registry));
    const auto start = std::chrono::steady_clock::now();
    manager->Initialize();
    const auto initialized = std::chrono::steady_clock::now();
    registry.ReportProgress("transport", true);
    manager->BeamOn(static_cast<int>(beam.size()));
    registry.ReportProgress("transport", true);
    const auto finished = std::chrono::steady_clock::now();
    // BeamOn 已同步所有 worker。这里只汇总小型清单，CSV 由 Python 合并。
    std::sort(registry.reports.begin(), registry.reports.end(),
              [](const auto& a, const auto& b) { return a.id < b.id; });
    size_t count = 0;
    for (const auto& item : registry.reports) count += item.events;
    if (count != beam.size() || registry.reports.size() != static_cast<size_t>(threads))
      throw std::runtime_error("Incomplete worker event accounting");
    std::ofstream metadata(registry.directory + "/transport.json");
    metadata << std::setprecision(15)
             << "{\"protocol\":2,\"actual_threads\":" << threads
             << ",\"mode\":\"" << (threads == 1 ? "serial" : "multithreaded")
             << "\",\"random_engine\":\"" << CLHEP::HepRandom::getTheEngine()->name()
             << "\",\"initialization_s\":" << std::chrono::duration<double>(initialized-start).count()
             << ",\"transport_s\":" << std::chrono::duration<double>(finished-initialized).count()
             << ",\"workers\":[";
    for (size_t i = 0; i < registry.reports.size(); ++i) {
      if (i) metadata << ',';
      metadata << "{\"id\":" << registry.reports[i].id
               << ",\"events\":" << registry.reports[i].events << '}';
    }
    metadata << "]}\n";
    metadata.close();
    if (!metadata) throw std::runtime_error("Cannot write transport metadata");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "droplet_g4: " << error.what() << '\n';
    return 1;
  }
}
